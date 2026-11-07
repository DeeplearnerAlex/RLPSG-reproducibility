"""Conservative redundant-row screen using existing actuator outer balls.

No neighbor search, geometry deletion, command choice or simulation occurs here.
For a stored pair row n.T @ (c_i-c_j) >= lower, every accepted command obeys
the existing speed and slew balls. The outer-ball directional lower bound
therefore certifies some solver rows redundant. Nonunit normals are supported.

The caller must retain the full original warning geometry and final matrix/
hard-cap verification. Floating-point guard is a screening margin, NEVER a
relaxation of the original acceptance tolerance. Dropping certified solver rows
preserves the mathematical feasible set, but finite solver iterates need not be
bitwise identical to an unpruned solve.
"""
from __future__ import annotations

import numpy as np


def redundant_pair_rows(normals, pairs, lower, v, reach, speed_cap,
                        acceptance_slack=1e-8, guard=1e-9):
    """Return a P-element bool mask of certified redundant lower-bound rows.

    ``pairs[k]=(i,j)`` and ``normals[k]`` describe the original ordered row.
    ``reach`` is tau*amax minus the per-robot velocity uncertainty; it is NOT
    enlarged when vmax increases. A scalar reach broadcasts to all N robots.
    Both supplied acceptance slack and guard must be finite and nonnegative.

    With S=speed_cap+slack, g_i=reach_i+slack, and z=||n||,
        L_i=max(-S*z, n.T*v_i-g_i*z),
        U_j=min( S*z, n.T*v_j+g_j*z).
    A row is screened only when L_i-U_j >= lower+guard. This includes commands
    in the acceptance-slack-expanded balls, not only strict actuator interiors.
    Empty pair arrays are accepted. Overflowing derived bounds are rejected.
    """
    normal = np.asarray(normals, dtype=np.float64)
    velocity = np.asarray(v, dtype=np.float64)
    pair = np.asarray(pairs)
    bound = np.asarray(lower, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError('v must have shape (N,2)')
    if pair.ndim != 2 or pair.shape[1] != 2 or pair.dtype.kind not in 'iu':
        raise ValueError('pairs must have integer shape (P,2)')
    count, population = len(pair), len(velocity)
    if normal.shape != (count, 2) or bound.shape != (count,):
        raise ValueError('normals and lower must have shape (P,2) and (P,)')
    if (not np.isfinite(velocity).all() or not np.isfinite(normal).all()
            or not np.isfinite(bound).all()):
        raise ValueError('States, normals and lower bounds must be finite')
    if (np.any(pair < 0) or np.any(pair >= population)
            or np.any(pair[:, 0] == pair[:, 1])):
        raise ValueError('Each pair must contain distinct valid robot indices')
    try:
        radii = np.asarray(reach, dtype=np.float64)
        if radii.shape not in [(), (population,)]:
            raise ValueError('Invalid reach shape')
        radii = np.broadcast_to(radii, (population,))
    except ValueError as error:
        raise ValueError('reach must be a scalar or N-element array') from error
    if not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError('reach must be finite and nonnegative')
    parameters = [np.asarray(x, dtype=np.float64) for x in
                  (speed_cap, acceptance_slack, guard)]
    if any(x.ndim != 0 or not np.isfinite(x) for x in parameters):
        raise ValueError('speed_cap, acceptance_slack and guard must be finite scalars')
    cap, slack, margin = map(float, parameters)
    if cap <= 0 or slack < 0 or margin < 0:
        raise ValueError('Positive speed_cap and nonnegative slack/guard required')
    if not count:
        return np.empty(0, dtype=bool)
    i, j = pair.T
    try:
        with np.errstate(over='raise', invalid='raise'):
            size = np.hypot(normal[:, 0], normal[:, 1])
            speed_extent = (cap+slack)*size
            dot_i = np.sum(normal*velocity[i], axis=1)
            dot_j = np.sum(normal*velocity[j], axis=1)
            minimum_i = np.maximum(-speed_extent, dot_i-(radii[i]+slack)*size)
            maximum_j = np.minimum(speed_extent, dot_j+(radii[j]+slack)*size)
            result = minimum_i-maximum_j >= bound+margin
    except FloatingPointError as error:
        raise ValueError('Derived directional bounds overflowed') from error
    return np.asarray(result, dtype=bool)


