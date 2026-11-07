"""Optional equivalent-computation helpers for the estimate-only controller.

No physical plant is run here. The original controller's function bytecode,
parameters, inequality assembly, solver sweep limits, acceptance tolerances,
fallback and uncertainty inputs are retained. Its private function-global map
is copied, not mutated, so importing this module cannot change an active caller
of nonideal_safety_controller. Only a conservative obstacle broad phase is added
to the existing estimated-trajectory warning check. Near circles still use the
original per-subarc cubic calculation and tolerance.

This is a computational variant, not a stronger nonideal safety guarantee.
The original controller module must be retained in its source snapshot.
"""
from __future__ import annotations

from types import FunctionType
import numpy as np

import nonideal_safety_controller as _original


def estimated_interval_clear_fast(p, v, command, model, duration, tube, radius,
                                  safe, side, circles, clearance):
    """Same estimated swept check, excluding provably unreachable circles.

    Every subarc velocity is affine; its norm is bounded by its endpoint norms.
    Consequently a robot travels at most maximum_endpoint_speed * total_time.
    A circle farther from the initial center than its required clearance plus
    that bound cannot fail the original distance test anywhere in the interval.
    An additional 1e-8 m broad-phase buffer makes exclusion conservative relative
    to, rather than looser than, the unchanged 1e-8 m narrow-phase tolerance.
    """
    from accelerated_arc_batch_metrics import arc_batch_metrics
    _, _, arcs = model.advance(p, v, command, duration)
    largest = float(tube.max())
    distance = safe + 2 * largest
    metrics = arc_batch_metrics(arcs, distance, robot_radius=radius+largest,
                                domain_side=side)
    if (metrics['minimum_separation_m'] < distance-1e-8 or
            metrics['minimum_boundary_margin_m'] < -1e-8):
        return False
    if not len(circles):
        return True
    speed = np.zeros(len(p), dtype=float)
    total_time = 0.
    for _, velocity, acceleration, h in arcs:
        speed = np.maximum(speed, np.linalg.norm(velocity, axis=1))
        speed = np.maximum(speed, np.linalg.norm(velocity+h*acceleration, axis=1))
        total_time += h
    displacement_bound = speed * total_time
    initial = arcs[0][0]
    for x, y, obstacle_radius in circles:
        center = np.array([x, y])
        required = obstacle_radius+radius+clearance+tube
        ids = np.flatnonzero(np.linalg.norm(initial-center, axis=1) <=
                             required+displacement_bound+1e-8)
        if not len(ids):
            continue
        for point, velocity, acceleration, h in arcs:
            distances = _original._arc_circle_distances(
                point[ids], velocity[ids], acceleration[ids], h, center)
            if (distances < required[ids]-1e-8).any():
                return False
    return True


_isolated_globals = dict(vars(_original))
_isolated_globals['_estimated_interval_clear'] = estimated_interval_clear_fast
filter_estimated = FunctionType(
    _original.filter_estimated.__code__, _isolated_globals,
    name='filter_estimated', argdefs=_original.filter_estimated.__defaults__,
    closure=_original.filter_estimated.__closure__)
filter_estimated.__kwdefaults__ = _original.filter_estimated.__kwdefaults__
filter_estimated.__annotations__ = dict(_original.filter_estimated.__annotations__)
filter_estimated.__doc__ = (
    'Original estimate-only controller with an isolated conservative circle '
    'broad phase in its unchanged estimated-subarc verification.')


