"""Least-change velocity commands with a relative, dynamically reachable backup.

All guarantees here concern the exact reduced plant, exact state and synchronous
updates only. A common damped-velocity backup replaces independent stopping
balls. The original PPO and nominal VOAPF commands are not retrained/rewritten.
"""
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from reduced_velocity_dynamics import accelerated_pair_metrics, accelerated_boundary_margin
from actuator_polygon_projection import project_actuator_commands


def backup_parameters(model, duration=.1, facets=8):
    steps = int(np.ceil(duration/model.integration_dt_s))
    h = duration/steps
    if h >= model.tau_s or facets < 4 or facets % 2:
        raise ValueError('Need h<tau and an even polygon with >=4 facets')
    q = (1-h/model.tau_s)**steps
    A = (model.tau_s-h/2)*(1-q)
    B = duration-A
    inscribed = np.cos(np.pi/facets)
    alpha = min(.5, model.tau_s*model.acceleration_cap_m_s2*inscribed/model.speed_cap_m_s)
    if alpha < 1-inscribed:
        raise ValueError('Use more polygon facets for this small tau*amax/vmax')
    damping = 1-alpha
    decay = 1-alpha*(1-q)
    K = (duration-alpha*B)/(1-decay)
    return dict(A=A, B=B, q=q, damping=damping, K=K, facets=facets,
                inscribed=inscribed, duration=duration, integration_h=h)


def segment_pairs(points, velocity, K, safe_distance, pairs=None):
    """Pair minima on the entire common backup segment, not endpoint distances."""
    if pairs is None:
        bound = K*float(np.linalg.norm(velocity, axis=1).max())
        pairs = cKDTree(points).query_pairs(safe_distance+2*bound+1e-8, output_type='ndarray')
    if not len(pairs):
        return pairs, np.empty((0, 2)), np.empty(0)
    i, j = pairs.T
    r, u = points[i]-points[j], velocity[i]-velocity[j]
    beta = np.clip(-np.sum(r*u, axis=1)/np.maximum(np.sum(u*u, axis=1), 1e-30), 0, K)
    closest = r+beta[:, None]*u
    distance = np.linalg.norm(closest, axis=1)
    return pairs, closest/np.maximum(distance[:, None], 1e-30), distance


def backup_certificate(points, velocity, model, duration=.1, safe_distance=1.05,
                       radius=.5, domain_side=800., facets=8):
    param = backup_parameters(model, duration, facets)
    pairs, normals, distances = segment_pairs(points, velocity, param['K'], safe_distance)
    stop = points+param['K']*velocity
    wall_min = min(points.min()-radius, stop.min()-radius,
                   domain_side-radius-points.max(), domain_side-radius-stop.max())
    pair_min = float(distances.min()) if len(distances) else float('inf')
    return pair_min >= safe_distance-1e-8 and wall_min >= -1e-8


def verified_command(points, velocity, command, model, duration=.1, safe_distance=1.05,
                     radius=.5, domain_side=800., facets=8):
    """Verify physical subarcs, actuator reachability, and complete backup tail."""
    if np.linalg.norm(command, axis=1).max() > model.speed_cap_m_s+1e-8:
        return False, {}
    if np.linalg.norm((command-velocity)/model.tau_s, axis=1).max() > model.acceleration_cap_m_s2+1e-8:
        return False, {}
    end, v_end, arcs = model.advance(points, velocity, command, duration)
    minimum, wall = float('inf'), float('inf')
    for p, v, a, dt in arcs:
        sep, _ = accelerated_pair_metrics(p, v, a, dt, safe_distance)
        minimum = min(minimum, sep)
        wall = min(wall, accelerated_boundary_margin(p, v, a, dt, radius, domain_side))
    ok = minimum >= safe_distance-1e-8 and wall >= -1e-8
    ok = ok and backup_certificate(end, v_end, model, duration, safe_distance, radius, domain_side, facets)
    return ok, dict(predicted_minimum_m=minimum, predicted_wall_margin_m=wall)


def relative_braking_filter(points, velocity, nominal_command, model, duration=.1,
                            safe_distance=1.05, radius=.5, domain_side=800., facets=8,
                            native_projection=False, support_verification=False,
                            projection_state=None, feasibility_restoration=False,
                            native_sweeps=800, restoration_sweeps=800):
    """min 0.5*sum ||c_i-c_nom_i||^2 with linear sufficient safety constraints.

    Inscribed polygons constrain ||c||<=vmax and ||c-v||<=tau*amax. The resulting
    actuator evolution is linear (saturation inactive). Pair support planes
    constrain both next positions and next damped-backup stopping points.
    The returned command is always checked using the actual plant. Solver or
    verification failure uses c_i=b*v_i, never an instantaneous velocity reset.
    """
    import osqp
    start = time.perf_counter()
    p, v, nominal = map(lambda x: np.asarray(x, dtype=float), (points, velocity, nominal_command))
    if p.shape != v.shape or p.shape != nominal.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 1:
        raise ValueError('Need matching N-by-2 state/command arrays, N>=1')
    if not np.isfinite([p, v, nominal]).all():
        raise ValueError('Nonfinite state/command')
    param = backup_parameters(model, duration, facets)
    def check_candidate(candidate):
        # Optional analytic certificate for the SAME held-command trajectory.
        # Keep the original cubic checker as the fallback and independent oracle.
        if support_verification:
            from relative_support_certificate import relative_support_certificate
            ok, detail = relative_support_certificate(p, v, candidate, model, duration,
                                                       safe_distance, radius, domain_side, facets)
            if ok:
                detail['verification_mode'] = 'support'
                return ok, detail
            # A failed sufficient support test alone must NOT reject a command.
            # However, the original verifier also requires the actual endpoint
            # backup. Check that necessary acceptance condition first, avoiding
            # expensive cubic roots for an already-invalid backup tail.
            end, v_end, _ = model.advance(p, v, candidate, duration)
            if not backup_certificate(end, v_end, model, duration,
                                      safe_distance, radius, domain_side, facets):
                return False, dict(verification_mode='endpoint_backup_rejection')
        ok, detail = verified_command(p, v, candidate, model, duration,
                                      safe_distance, radius, domain_side, facets)
        detail['verification_mode'] = 'cubic'
        return ok, detail

    def repair_actuator_roundoff(candidate):
        # An inner polygon's vertices can exceed the circle by roundoff. A tiny
        # blend toward the reachable backup is explicit, not a state reset.
        if not support_verification:
            return candidate
        speed = np.linalg.norm(candidate, axis=1).max()
        slew = np.linalg.norm(candidate-v, axis=1).max()
        if speed <= model.speed_cap_m_s and slew <= model.tau_s*model.acceleration_cap_m_s2:
            return candidate
        if speed <= model.speed_cap_m_s+1e-7 and slew <= model.tau_s*model.acceleration_cap_m_s2+1e-7:
            return param['damping']*v+(1-1e-6)*(candidate-param['damping']*v)
        return candidate

    info = dict(method='relative braking least-change QP', prediction_checks=0,
                braking_robots=0, all_braking=False, feasible=False,
                qp_solved=False, backup_fallback=False, backup_horizon_factor_s=param['K'])
    if not backup_certificate(p, v, model, duration, safe_distance, radius, domain_side, facets):
        info.update(reason='outside_sufficient_relative_backup_domain', seconds=time.perf_counter()-start)
        return None, info
    n = len(p)
    A, B, q, K = (param[key] for key in ['A', 'B', 'q', 'K'])
    zv, zc = A+K*q, B+K*(1-q)
    angle = np.arange(facets)*2*np.pi/facets
    normals_polygon = np.column_stack((np.cos(angle), np.sin(angle)))
    cap = param['inscribed']
    speed_limit = model.speed_cap_m_s*cap
    acceleration_limit = model.tau_s*model.acceleration_cap_m_s2*cap
    nominal_poly = nominal@normals_polygon.T
    delta_poly = (nominal-v)@normals_polygon.T
    if nominal_poly.max() <= speed_limit and delta_poly.max() <= acceleration_limit:
        info['prediction_checks'] += 1
        ok, detail = check_candidate(nominal)
        if ok:
            info.update(detail, feasible=True, reason='nominal_feasible', seconds=time.perf_counter()-start)
            return nominal.copy(), info
    # If the independent actuator-optimal projection is already physically safe,
    # no coupled QP is needed. This is not a heuristic norm contraction.
    projected = project_actuator_commands(nominal, v, model.tau_s,
                                          model.acceleration_cap_m_s2, model.speed_cap_m_s, facets)
    projected = repair_actuator_roundoff(projected)
    info['prediction_checks'] += 1
    ok, detail = check_candidate(projected)
    if ok:
        info.update(detail, feasible=True, reason='actuator_projection_verified',
                    braking_robots=int((np.linalg.norm(projected-nominal, axis=1)>1e-6).sum()),
                    seconds=time.perf_counter()-start)
        return projected, info
    cutoff = safe_distance+2*K*model.speed_cap_m_s+2*model.speed_cap_m_s*duration+1e-8
    pairs = cKDTree(p).query_pairs(cutoff, output_type='ndarray')
    pairs, normal, distance = segment_pairs(p, v, K, safe_distance, pairs)
    blocks, lower, upper = [], [], []
    if len(pairs):
        i, j = pairs.T
        r, u = p[i]-p[j], v[i]-v[j]
        bound_z = (safe_distance-np.sum(normal*(r+zv*u), axis=1))/zc
        bound_p = (safe_distance-np.sum(normal*(r+A*u), axis=1))/B
        pair_matrix = sparse.csc_matrix((np.column_stack((normal, -normal)).ravel(),
                                        (np.repeat(np.arange(len(pairs)), 4),
                                         np.column_stack((2*i, 2*i+1, 2*j, 2*j+1)).ravel())),
                                       shape=(len(pairs), 2*n))
        blocks.append(pair_matrix)
        lower.append(np.maximum(bound_z, bound_p))
        upper.append(np.full(len(pairs), np.inf))
    polygon = sparse.kron(sparse.eye(n, format='csc'), sparse.csc_matrix(normals_polygon), format='csc')
    blocks.append(polygon)
    lower.append(np.full(n*facets, -np.inf))
    upper.append(np.minimum(speed_limit, (v@normals_polygon.T).ravel()+acceleration_limit))
    # End position and backup point must both remain in the rectangular world.
    low = np.maximum((radius-p-A*v)/B, (radius-p-zv*v)/zc).ravel()
    high = np.minimum((domain_side-radius-p-A*v)/B, (domain_side-radius-p-zv*v)/zc).ravel()
    active_walls = (low > -model.speed_cap_m_s) | (high < model.speed_cap_m_s)
    if active_walls.any():
        blocks.append(sparse.eye(2*n, format='csc')[active_walls])
        lower.append(low[active_walls]); upper.append(high[active_walls])
    matrix = sparse.vstack(blocks, format='csc')
    row_keys = np.concatenate([
        pairs[:, 0]*n+pairs[:, 1],
        n*n+np.arange(n*facets, dtype=np.int64),
        n*n+n*facets+np.flatnonzero(active_walls),
    ]).astype(np.int64, copy=False)
    # The damped backup is a known dynamically reachable feasible initial guess.
    backup = param['damping']*v
    if native_projection:
        try:
            lo, hi = np.concatenate(lower), np.concatenate(upper)
            native_dual = None
            if projection_state is None:
                from command_projection import project_command_polytope
                projected_qp, native_info = project_command_polytope(nominal, matrix,
                                                                    lo, hi, max_sweeps=native_sweeps)
            else:
                from warm_command_projection import project_command_polytope_warm
                mapped = np.zeros(len(row_keys))
                reused = 0
                if projection_state.get('population') == n and len(projection_state.get('row_keys', [])):
                    old_keys, old_dual = projection_state['row_keys'], projection_state['dual']
                    order = np.argsort(old_keys)
                    sorted_keys = old_keys[order]
                    index = np.searchsorted(sorted_keys, row_keys)
                    match = index < len(sorted_keys)
                    match[match] &= sorted_keys[index[match]] == row_keys[match]
                    mapped[match] = old_dual[order[index[match]]]
                    reused = int(match.sum())
                # Retain only dual values in the CURRENT slab's support domain.
                mapped[np.isposinf(hi) & (mapped > 0)] = 0.
                mapped[np.isneginf(lo) & (mapped < 0)] = 0.
                projected_qp, dual, native_info = project_command_polytope_warm(
                    nominal, matrix, lo, hi, dual=mapped, max_sweeps=native_sweeps)
                native_dual = dual.copy()
                native_info['warm_rows_reused'] = reused
                if np.isfinite(dual).all():
                    projection_state.update(population=n, row_keys=row_keys.copy(), dual=dual.copy())
            info['native_projection'] = native_info
            restoration_center = projected_qp.copy()
            if native_info['primal_residual'] <= 1e-8:
                projected_qp = repair_actuator_roundoff(projected_qp)
                info['prediction_checks'] += 1
                ok, detail = check_candidate(projected_qp)
                if ok:
                    info.update(detail, feasible=True, qp_solved=bool(native_info['converged']),
                                reason='native_projection_verified',
                                braking_robots=int((np.linalg.norm(projected_qp-nominal, axis=1)>1e-6).sum()),
                                seconds=time.perf_counter()-start)
                    return projected_qp, info
            if feasibility_restoration and np.isfinite(restoration_center).all():
                # This SECOND projection repairs near-feasibility, not the
                # original objective. Never label it the original QP optimum.
                from warm_command_projection import project_command_polytope_warm
                restored, repair_dual, repair_info = project_command_polytope_warm(
                    restoration_center, matrix, lo, hi, dual=None,
                    max_sweeps=restoration_sweeps)
                info['native_restoration'] = repair_info
                restored = repair_actuator_roundoff(restored)
                info['prediction_checks'] += 1
                ok, detail = check_candidate(restored)
                if ok:
                    if (projection_state is not None and native_dual is not None
                            and native_dual.shape == repair_dual.shape == row_keys.shape):
                        with np.errstate(over='ignore', invalid='ignore'):
                            combined_dual = native_dual+repair_dual
                        if np.isfinite(combined_dual).all():
                            projection_state.update(population=n, row_keys=row_keys.copy(),
                                                    dual=combined_dual)
                    info.update(detail, feasible=True, qp_solved=False,
                                reason='native_feasibility_restoration_verified',
                                restoration_maximum_command_change_m_s=float(
                                    np.linalg.norm(restored-restoration_center, axis=1).max()),
                                braking_robots=int((np.linalg.norm(restored-nominal, axis=1)>1e-6).sum()),
                                seconds=time.perf_counter()-start)
                    return restored, info
        except Exception as error:
            info['native_projection_error'] = f'{type(error).__name__}: {error}'
    solution = None
    solver_status = 'not_run'
    try:
        problem = osqp.OSQP()
        problem.setup(P=sparse.eye(2*n, format='csc'), q=-nominal.ravel(), A=matrix,
                      l=np.concatenate(lower), u=np.concatenate(upper), verbose=False,
                      eps_abs=2e-8, eps_rel=2e-8, max_iter=4000, polishing=True)
        problem.warm_start(x=backup.ravel())
        solution = problem.solve(raise_error=False)
        solver_status = solution.info.status
        # A finite iteration/time-limited dual is still a valid next warm start:
        # the next call rebuilds x0 around its CURRENT nominal and CURRENT A.
        if projection_state is not None and solution.info.status_val in [1, 2, 7, 8] and solution.y is not None and np.isfinite(solution.y).all():
            projection_state.update(population=n, row_keys=row_keys.copy(), dual=solution.y.copy())
    except Exception as error:
        solver_status = f'{type(error).__name__}: {error}'
    if solution is not None and solution.x is not None and np.isfinite(solution.x).all() and solution.info.status_val in [1, 2, 7, 8]:
        optimal = solution.info.status_val in [1, 2]
        candidate = repair_actuator_roundoff(solution.x.reshape(n, 2))
        info['prediction_checks'] += 1
        ok, detail = check_candidate(candidate)
        if ok:
            info.update(detail, feasible=True, qp_solved=optimal, reason='qp_verified',
                        solver_status=solver_status, finite_iterate_accepted=not optimal,
                        braking_robots=int((np.linalg.norm(candidate-nominal, axis=1)>1e-6).sum()),
                        seconds=time.perf_counter()-start)
            return candidate, info
        # Numerical repair stays inside the convex segment toward the known
        # dynamically reachable backup, never toward an instantaneous stop.
        for fraction in [.999, .99, .9, .5]:
            repaired = backup+fraction*(candidate-backup)
            info['prediction_checks'] += 1
            ok, detail = check_candidate(repaired)
            if ok:
                info.update(detail, feasible=True, qp_solved=optimal, reason='qp_blend_verified',
                            solver_status=solver_status, finite_iterate_accepted=not optimal,
                            qp_blend_fraction=fraction,
                            braking_robots=int((np.linalg.norm(repaired-nominal, axis=1)>1e-6).sum()),
                            seconds=time.perf_counter()-start)
                return repaired, info
    info['prediction_checks'] += 1
    ok, detail = check_candidate(backup)
    info.update(detail, feasible=ok, backup_fallback=True, all_braking=True, braking_robots=n,
                solver_status=solver_status, seconds=time.perf_counter()-start,
                reason='verified_damped_backup' if ok else 'damped_backup_failed_numeric_check')
    return backup if ok else None, info

