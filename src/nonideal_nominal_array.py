"""Array-only execution of the SAME speed-aware nominal command arithmetic.

This optional planner omits only Robot construction, copying and unused distance
bookkeeping. It keeps the original observation-only repulsion, step-start mode
selection, bin-group insertion order, native/SciPy centroid callback, duplicate
coordinate dictionary lookup, threshold gray checks and floating operations.
It does not replace the executed plant, allocation, obstacle or safety layers.
Only the float64 frames emitted by CombinedObservationNetwork are supported.
"""
from __future__ import annotations

import numpy as np

from native_voronoi_centroids import native_voronoi_centroids
from reduced_velocity_dynamics import norm_clip
from motion_utils import DT, sensed_repulsion, bins_and_counts


class ArrayEstimatedNominal:
    """Equivalent command/stop outputs to SpeedAwareEstimatedNominal.

    There are deliberately no Robot objects: intermediate virtual positions and
    velocities are arrays, and only commands/stop are consumed by the runner.
    ``targets`` remains the same last-write-wins coordinate dictionary, including
    the unusual duplicate-position case in which an unreached robot finds a
    centroid recorded for another, reached robot at the identical coordinate.
    """
    def __init__(self):
        self.frame = self.goals = None
        self.targets = {}
        self._trial_speed_cap = None

    def __call__(self, frame, goals, model):
        cap = float(model.speed_cap_m_s)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError('A finite positive model speed cap is required')
        if self._trial_speed_cap is not None and self._trial_speed_cap != cap:
            raise ValueError('Changing speed cap within a nominal-planner trial is not allowed')
        self._trial_speed_cap = cap
        p, v = np.asarray(frame.p_hat), np.asarray(frame.v_hat)
        goals = np.asarray(goals)
        if (p.dtype != np.float64 or v.dtype != np.float64 or p.ndim != 2
                or p.shape[1:] != (2,) or v.shape != p.shape or not len(p)
                or goals.shape != (len(p),) or not np.issubdtype(goals.dtype, np.integer)
                or not np.isfinite([p, v]).all() or np.any((goals < 0) | (goals >= 64))):
            raise ValueError('Expected float64 observed states and valid integer goal bins')
        self.frame = frame
        self.goals = goals.copy()
        self.targets = {}

        # These are exactly the Robot goal-center and cell-bound arrays, not a
        # changed binning rule. set_actual_state resets every stop flag to False.
        centers = (np.column_stack((goals % 8, goals//8))*100+50).astype(float)
        low = np.column_stack((goals % 8, goals//8))*100.
        high = low+100.
        reached_before = bins_and_counts(p)[0] == goals
        reached = reached_before.copy()
        stop = np.zeros(len(p), dtype=bool)
        force = sensed_repulsion(frame, len(p), reached_before, 3.)
        velocity = v.copy()
        after = p.copy()

        unreached = np.flatnonzero(~reached_before)
        if len(unreached):
            inside = np.all((p[unreached] >= low[unreached])
                            & (p[unreached] < high[unreached]), axis=1)
            reached[unreached] |= inside
            attractive = 1.*(centers[unreached]-p[unreached])
            attractive[reached[unreached]] = 0.
            velocity[unreached] += (attractive+force[unreached])*DT
            speeds = np.linalg.norm(velocity[unreached], axis=1)
            limited = speeds > cap
            rows = unreached[limited]
            if len(rows):
                velocity[rows] = velocity[rows]/speeds[limited, None]*cap
            after[unreached] = np.clip(p[unreached]+velocity[unreached]*DT,
                                       [1e-6, 1e-6], np.asarray((800, 800))-1e-6)

        # defaultdict insertion order is the first reached identity per bin;
        # within-bin point ordering follows the original robot identity order.
        groups = {}
        for index in np.flatnonzero(reached_before):
            groups.setdefault(int(goals[index]), []).append(index)
        decisions = (.1, 1.) if cap == 1. else (.1, 2*model.tau_s*cap)
        for group in groups.values():
            ids = np.asarray(group)
            first = group[0]
            targets = native_voronoi_centroids(
                p[ids], low[first, 0], high[first, 0], low[first, 1], high[first, 1],
                decision_distances=decisions)
            for point, target in zip(p[ids], targets):
                self.targets[tuple(point)] = target
            error = targets-p[ids]
            remaining = np.linalg.norm(error, axis=1)
            moving = remaining > .1
            velocity[ids] = 0.
            velocity[ids[moving]] = error[moving]/remaining[moving, None]*1.
            stop[ids] = ~moving
            after[ids] = np.clip(p[ids]+velocity[ids]*DT, low[ids]+1e-6, high[ids]-1e-6)

        command = (after-p)/DT
        error = np.zeros_like(p)
        has_target = np.zeros(len(p), dtype=bool)
        for i, point in enumerate(p):
            target = self.targets.get(tuple(point))
            if target is not None:
                error[i] = target-point
                has_target[i] = True
        distance = np.linalg.norm(error, axis=1)
        threshold = 2*model.tau_s*cap
        gray = has_target & ((abs(distance-.1) <= 1e-10) | (abs(distance-threshold) <= 1e-10))
        for i in np.flatnonzero(gray):
            distance[i] = np.linalg.norm(error[i])
        if cap == 1.:
            near = has_target & (distance <= threshold)
        else:
            far = has_target & (distance > threshold)
            command[far] = error[far]/distance[far, None]*cap
            near = has_target & ~far
        if near.any():
            proposed = error[near]/model.tau_s-v[near]
            arrived = distance[near] <= .1
            proposed[arrived] = -v[near][arrived]
            command[near] = norm_clip(proposed, cap)
        if cap != 1.:
            if has_target.any():
                ids = np.flatnonzero(has_target)
                lower = np.column_stack((goals[ids] % 8, goals[ids]//8))*100+1e-6
                upper = lower+100-2e-6
                intended = p[ids]+DT*command[ids]
                clipped = np.clip(intended, lower, upper)
                changed = np.any(clipped != intended, axis=1)
                command[ids[changed]] = (clipped[changed]-p[ids[changed]])/DT
            command = norm_clip(command, cap)
        return command, bool(stop.all())

