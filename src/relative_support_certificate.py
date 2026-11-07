"""Fast sufficient convex-support check for the exact reduced lagged model.

This is a floating-point sufficient check, not a formal numerical certificate or
a result for complete UAV dynamics. False means "use the original verifier",
not necessarily that the command is unsafe. Current state, model and commands
must obey strict speed/reachability bounds; uncertain saturation cases fall back.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def relative_support_certificate(points, velocity, command, model, duration=.1,
                                 safe_distance=1.05, radius=.5, domain_side=800., facets=8):
    """Check four supporting vertices for each potentially interacting pair.

    For a held reachable command, let P=p+A*v+B*c, W=q*v+(1-q)*c, Z=P+K*W.
    A(s) is increasing and concave for the unsaturated substep execution model,
    and B(s)=s-A(s). Set beta=B(s)/B(T), delta=A(s)-beta*A(T). If K>=A(T),
    then delta>=0 and delta/K<=1-beta, so every physical trajectory point lies
    in conv{p, p+K*v, P}. The same holds for relative pair trajectories. The
    common damped backup occupies the segment P--Z. A single unit normal that
    supports all four pair-relative vertices by d_safe therefore suffices.

    Distant pairs are excluded only using d+2*K*vmax+2*T*vmax+1e-8. Their
    distance remains above d for both the physical interval and backup. Walls
    are checked at every robot's four absolute vertices. All statements are
    conditional on this exact reduced plant and synchronous held commands.
    """
    diagnostics = dict(method='four-vertex relative convex-support sufficient check',
                       ok=False, checked_pairs=0, minimum_checked_pair_plane_m=None,
                       minimum_checked_wall_margin_m=None, support_tolerance_m=1e-9)

    def reject(reason):
        diagnostics['reason'] = reason
        return False, diagnostics

    try:
        p, v, c = (np.asarray(a, dtype=np.float64) for a in (points, velocity, command))
    except (TypeError, ValueError):
        return reject('invalid_state_array')
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or v.shape != p.shape or c.shape != p.shape:
        return reject('invalid_state_shape')
    if not np.isfinite([p, v, c]).all():
        return reject('nonfinite_state_or_command')
    scalars = [duration, safe_distance, radius, domain_side, model.tau_s,
               model.speed_cap_m_s, model.acceleration_cap_m_s2, model.integration_dt_s]
    if not np.isfinite(scalars).all() or min(scalars) <= 0 or domain_side <= 2*radius:
        return reject('invalid_model_or_geometry')
    if not isinstance(facets, (int, np.integer)) or facets < 4 or facets % 2:
        return reject('invalid_polygon_facets')
    speed_v = np.linalg.norm(v, axis=1)
    speed_c = np.linalg.norm(c, axis=1)
    slew = np.linalg.norm(c-v, axis=1)
    vmax = float(model.speed_cap_m_s)
    slew_cap = float(model.tau_s*model.acceleration_cap_m_s2)
    diagnostics.update(maximum_current_speed_m_s=float(speed_v.max()),
                       maximum_command_speed_m_s=float(speed_c.max()),
                       maximum_command_slew_m_s=float(slew.max()))
    # No +tol here: any possible saturation invalidates the affine proof and
    # must be handled by the full, saturating execution-model verifier instead.
    if speed_v.max() > vmax or speed_c.max() > vmax:
        return reject('strict_speed_bound_not_satisfied')
    if slew.max() > slew_cap:
        return reject('strict_reachability_bound_not_satisfied')
    try:
        # Lazy import avoids an import cycle when the filter uses this helper.
        from relative_braking_filter import backup_parameters
        param = backup_parameters(model, duration, facets)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return reject('backup_parameters_unavailable')
    A, B, q, K = (float(param[key]) for key in ('A', 'B', 'q', 'K'))
    if not np.isfinite([A, B, q, K]).all() or B <= 0 or A < 0 or K < A or not 0 <= q <= 1:
        return reject('convex_trajectory_conditions_not_satisfied')
    diagnostics.update(A_s=A, B_s=B, q=q, backup_K_s=K)
    current_stop = p+K*v
    next_position = p+A*v+B*c
    next_velocity = q*v+(1-q)*c
    next_stop = next_position+K*next_velocity
    wall_margin = min(float(vertex.min()-radius) for vertex in (p, current_stop, next_position, next_stop))
    wall_margin = min(wall_margin, min(float(domain_side-radius-vertex.max())
                                     for vertex in (p, current_stop, next_position, next_stop)))
    diagnostics['minimum_checked_wall_margin_m'] = wall_margin
    if wall_margin < -1e-9:
        return reject('wall_support_not_satisfied')
    cutoff = safe_distance+2*K*vmax+2*duration*vmax+1e-8
    pairs = cKDTree(p).query_pairs(cutoff, output_type='ndarray')
    diagnostics.update(checked_pairs=len(pairs), pair_cutoff_m=float(cutoff),
                       omitted_pairs_safe_by_displacement_bound=True)
    if len(pairs):
        i, j = pairs.T
        r = p[i]-p[j]
        u = v[i]-v[j]
        dc = c[i]-c[j]
        beta = np.clip(-np.sum(r*u, axis=1)/np.maximum(np.sum(u*u, axis=1), 1e-30), 0., K)
        nearest = r+beta[:, None]*u
        normal = nearest/np.maximum(np.linalg.norm(nearest, axis=1, keepdims=True), 1e-30)
        # Re-evaluate all geometric supports directly; do not use a solver's
        # residual estimate or the divided/merged QP constraint RHS values.
        relative_stop = r+K*u
        relative_next = r+A*u+B*dc
        relative_next_stop = relative_next+K*(q*u+(1-q)*dc)
        supports = np.column_stack([np.sum(normal*vertex, axis=1)
                                    for vertex in (r, relative_stop, relative_next, relative_next_stop)])
        minimum = float(supports.min())
        diagnostics['minimum_checked_pair_plane_m'] = minimum
        if minimum < safe_distance-1e-9:
            return reject('pair_support_not_satisfied')
    diagnostics.update(ok=True, reason='relative_convex_support_satisfied')
    return True, diagnostics

