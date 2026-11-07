"""Exact 2D Euclidean projection onto merged actuator command polygons.

The word exact refers to exhaustive convex-polygon candidate enumeration, not
exact arithmetic: the implementation uses float64 and a stated feasibility
tolerance. No pairwise safety or braking constraints are handled here.
"""
from __future__ import annotations

import numpy as np


def _parameters(tau, acceleration_cap, speed_cap, facets):
    if not isinstance(facets, (int, np.integer)) or facets < 4 or facets % 2:
        raise ValueError('facets must be an even integer >= 4')
    if not np.isfinite([tau, acceleration_cap, speed_cap]).all() or min(tau, acceleration_cap, speed_cap) <= 0:
        raise ValueError('tau, acceleration_cap and speed_cap must be finite and positive')
    angle = np.arange(facets)*2*np.pi/facets
    normal = np.column_stack((np.cos(angle), np.sin(angle)))
    rho = np.cos(np.pi/facets)
    return normal, rho


def actuator_polygon_bounds(velocity, tau=.5, acceleration_cap=1., speed_cap=1., facets=8):
    """Return U and b for U c_i <= b_i, shape U=(M,2), b=(N,M).

    Speed rows: U c_i <= speed_cap*cos(pi/M).
    Slew rows: U(c_i-v_i) <= tau*acceleration_cap*cos(pi/M).
    Their parallel identical normals merge exactly by taking the smaller RHS.
    """
    normal, rho = _parameters(tau, acceleration_cap, speed_cap, facets)
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 2 or not np.isfinite(velocity).all():
        raise ValueError('velocity must be a finite N-by-2 array')
    bound = np.minimum(speed_cap*rho, velocity@normal.T + tau*acceleration_cap*rho)
    return normal, bound


def project_actuator_commands(nominal, velocity, tau=.5, acceleration_cap=1., speed_cap=1., facets=8,
                              tolerance=1e-10):
    """Project each nominal 2D command onto its convex actuator polygon.

    Returns an independent N-by-2 float64 array, without changing input arrays.
    A closest point is the feasible nominal itself, an orthogonal projection
    onto a face, or a vertex formed by two nonparallel faces. Enumerating all
    such candidates therefore solves each two-dimensional projection problem.
    Parallel face pairs need no vertices; degenerate feasible faces are still
    covered by face projections and intersections with their other faces.

    Raises ValueError if at least one row has no feasible candidate (empty or
    numerically unresolved polygon). This function is not a safety filter.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if nominal.shape != velocity.shape or nominal.ndim != 2 or nominal.shape[1] != 2:
        raise ValueError('nominal and velocity must have matching N-by-2 shapes')
    if not np.isfinite(nominal).all() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nominal commands and finite nonnegative tolerance required')
    normal, bound = actuator_polygon_bounds(velocity, tau, acceleration_cap, speed_cap, facets)
    result = nominal.copy()
    if not len(nominal):
        return result
    outside = np.any(nominal@normal.T > bound+tolerance, axis=1)
    indices = np.flatnonzero(outside)
    if not len(indices):
        return result
    desired = nominal[indices]
    limits = bound[indices]

    # Every normal is unit length, so the face projection needs no division.
    excess = desired@normal.T-limits
    faces = desired[:, None, :]-excess[:, :, None]*normal[None]

    first, second = np.triu_indices(facets, 1)
    determinant = normal[first, 0]*normal[second, 1]-normal[first, 1]*normal[second, 0]
    nonparallel = abs(determinant) > 1e-12
    first, second, determinant = first[nonparallel], second[nonparallel], determinant[nonparallel]
    x = (limits[:, first]*normal[second, 1]-normal[first, 1]*limits[:, second])/determinant
    y = (normal[first, 0]*limits[:, second]-limits[:, first]*normal[second, 0])/determinant
    vertices = np.stack((x, y), axis=2)
    candidates = np.concatenate((faces, vertices), axis=1)

    evaluated = np.einsum('nkd,md->nkm', candidates, normal)
    feasible = np.all(evaluated <= limits[:, None, :]+tolerance, axis=2)
    distance_squared = np.sum((candidates-desired[:, None, :])**2, axis=2)
    distance_squared = np.where(feasible, distance_squared, np.inf)
    selected = np.argmin(distance_squared, axis=1)
    valid = np.isfinite(distance_squared[np.arange(len(indices)), selected])
    if not valid.all():
        bad = indices[~valid].tolist()
        raise ValueError(f'No feasible actuator-polygon projection for rows {bad[:16]}')
    result[indices] = candidates[np.arange(len(indices)), selected]
    return result

