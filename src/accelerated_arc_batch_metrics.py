"""One-neighbor-list swept checks for a complete piecewise-parabolic interval.

No sample times or extrema are dropped. Candidate pairs use a conservative
whole-interval travel bound; every retained pair is checked on every subarc
using the same cubic derivative roots and numerical overlap tolerance as the
existing per-arc checker. Inputs must form one continuous physical trajectory.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def arc_batch_metrics(arcs, contact_distance, robot_radius=.5, domain_side=800.,
                      continuity_tolerance=1e-8):
    """Return separation/contact/wall/speed/acceleration/travel diagnostics.

    Each arc is (p_start, v_start, constant_acceleration, duration). The contact
    definition remains distance < contact_distance - 1e-8. Travel uses the same
    three-point Gaussian speed quadrature as the existing simulation runner.
    Distances and parabolic coordinate extrema are analytic up to float64 root
    arithmetic, not interval-certified. This does not change the motion model.
    """
    arcs = list(arcs)
    if not arcs:
        raise ValueError('At least one continuous motion arc is required')
    if not np.isfinite([contact_distance, robot_radius, domain_side, continuity_tolerance]).all():
        raise ValueError('Finite geometry and tolerance required')
    if contact_distance <= 0 or robot_radius < 0 or domain_side <= 2*robot_radius or continuity_tolerance < 0:
        raise ValueError('Invalid contact distance, geometry or continuity tolerance')
    p = np.stack([np.asarray(arc[0], dtype=float) for arc in arcs])
    v = np.stack([np.asarray(arc[1], dtype=float) for arc in arcs])
    a = np.stack([np.asarray(arc[2], dtype=float) for arc in arcs])
    duration = np.asarray([arc[3] for arc in arcs], dtype=float)
    if p.shape != v.shape or p.shape != a.shape or p.ndim != 3 or p.shape[2] != 2 or p.shape[1] < 1:
        raise ValueError('Motion arcs need matching, nonempty N-by-2 arrays')
    if not np.isfinite([p, v, a]).all() or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError('Finite states and positive subarc durations required')
    h = duration[:, None, None]
    end = p+h*v+.5*h*h*a
    end_velocity = v+h*a
    if len(arcs) > 1:
        mismatch = max(float(abs(end[:-1]-p[1:]).max()), float(abs(end_velocity[:-1]-v[1:]).max()))
        if mismatch > continuity_tolerance:
            raise ValueError(f'Input arcs are not contiguous: mismatch={mismatch}')

    speed = np.linalg.norm(v, axis=2)
    end_speed = np.linalg.norm(end_velocity, axis=2)
    speed_bound = float(max(speed.max(), end_speed.max()))
    acceleration_bound = float(np.linalg.norm(a, axis=2).max())
    total_time = float(duration.sum())
    n = p.shape[1]
    if n < 2:
        initial_minimum = float('inf')
        pairs = np.empty((0, 2), dtype=int)
    else:
        tree = cKDTree(p[0])
        initial_minimum = float(tree.query(p[0], k=2)[0][:, 1].min())
        # Each speed norm is convex on a constant-acceleration arc, so all path
        # speeds are bounded by the maximum of the velocity endpoint norms.
        cutoff = max(initial_minimum, contact_distance)+2*speed_bound*total_time+1e-8
        pairs = tree.query_pairs(cutoff, output_type='ndarray')
    minimum = initial_minimum
    contacts = np.empty((0, 2), dtype=int)
    if len(pairs):
        i, j = pairs.T
        r, u, b = p[:, i]-p[:, j], v[:, i]-v[:, j], .5*(a[:, i]-a[:, j])
        t_end = duration[:, None, None]
        r_end = r+t_end*u+t_end*t_end*b
        # Advanced pair indexing can produce Fortran-ordered intermediates;
        # force C order so the flattened root updates write into this array.
        squared = np.ascontiguousarray(np.minimum(np.sum(r*r, axis=2), np.sum(r_end*r_end, axis=2)))
        c0 = np.sum(r*u, axis=2)
        c1 = np.sum(u*u+2*r*b, axis=2)
        c2 = 3*np.sum(u*b, axis=2)
        c3 = 2*np.sum(b*b, axis=2)
        h_pair = duration[:, None]
        variation = abs(c1)*h_pair+abs(c2)*h_pair*h_pair+abs(c3)*h_pair*h_pair*h_pair
        possible_roots = (abs(c0) <= variation+1e-20) & ((abs(c1)+abs(c2)+abs(c3)) > 0)
        r_flat, u_flat, b_flat = (x.reshape(-1, 2) for x in (r, u, b))
        coefficient = [x.ravel() for x in (c3, c2, c1, c0)]
        squared_flat = squared.ravel()
        pair_count = len(pairs)
        for k in np.flatnonzero(possible_roots.ravel()):
            polynomial = np.trim_zeros(np.asarray([x[k] for x in coefficient]), 'f')
            if len(polynomial) < 2:
                continue
            roots = np.roots(polynomial)
            limit = duration[k//pair_count]
            times = roots.real[(abs(roots.imag) < 1e-9) & (roots.real > 0) & (roots.real < limit)]
            if len(times):
                curve = r_flat[k]+times[:, None]*u_flat[k]+times[:, None]**2*b_flat[k]
                squared_flat[k] = min(squared_flat[k], float(np.sum(curve*curve, axis=1).min()))
        distances = np.sqrt(squared)
        minimum = min(minimum, float(distances.min()))
        contacts = pairs[np.any(distances < contact_distance-1e-8, axis=0)]
        if len(contacts):
            contacts = contacts[np.lexsort((contacts[:, 1], contacts[:, 0]))]

    # Coordinate extrema on every parabolic subarc, not just controller ends.
    low, high = np.minimum(p, end), np.maximum(p, end)
    turning_time = np.divide(-v, a, out=np.zeros_like(v), where=abs(a)>1e-30)
    turning = (abs(a)>1e-30) & (turning_time > 0) & (turning_time < h)
    stationary = p+turning_time*v+.5*turning_time*turning_time*a
    low = np.where(turning, np.minimum(low, stationary), low)
    high = np.where(turning, np.maximum(high, stationary), high)
    wall_margin = float(min(low.min()-robot_radius, domain_side-robot_radius-high.max()))

    # Keep the existing runner's travel quadrature exactly, just batch the arcs.
    nodes = np.array([.5-np.sqrt(15)/10, .5, .5+np.sqrt(15)/10])
    weights = np.array([5/18, 4/9, 5/18])
    quadrature_time = duration[:, None]*nodes[None]
    quadrature_velocity = v[:, :, None, :]+a[:, :, None, :]*quadrature_time[:, None, :, None]
    travel = float((np.linalg.norm(quadrature_velocity, axis=3)*duration[:, None, None]*weights[None, None]).sum())
    return dict(minimum_separation_m=minimum, contact_pairs=contacts,
                minimum_boundary_margin_m=wall_margin,
                maximum_acceleration_m_s2=acceleration_bound, maximum_speed_m_s=speed_bound,
                travel_m=travel, candidate_pairs=len(pairs), arcs=len(arcs),
                duration_s=total_time)

