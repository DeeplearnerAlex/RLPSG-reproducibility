"""Opt-in state-only matched prediction; never use as the truth-world model.

Identical arithmetic to VelocityResponse.advance, omitting only the three
discarded per-substep arc copies. Network/belief prediction never consumes arcs.
The world/independent safety audit MUST retain the original arc-producing model.
There are no random draws, sensor decisions, timing changes or state corrections.
"""
from dataclasses import asdict

import numpy as np

from reduced_velocity_dynamics import VelocityResponse, norm_clip


class StateOnlyVelocityResponse(VelocityResponse):
    def advance(self, positions, velocities, command, duration_s):
        p, v, command = map(lambda x: np.array(x, dtype=float, copy=True), (positions, velocities, command))
        if p.shape != v.shape or command.shape != p.shape or p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Expected equal N-by-2 arrays')
        if not len(p) or not np.isfinite(duration_s) or not np.isfinite([p, v, command]).all() or duration_s <= 0:
            raise ValueError('Finite states and positive duration required')
        if np.linalg.norm(v, axis=1).max() > self.speed_cap_m_s+1e-9:
            raise ValueError('Initial velocity exceeds speed cap')
        command = norm_clip(command, self.speed_cap_m_s)
        steps = int(np.ceil(duration_s/self.integration_dt_s))
        h = duration_s/steps
        for _ in range(steps):
            acceleration = norm_clip((command-v)/self.tau_s, self.acceleration_cap_m_s2)
            next_v = norm_clip(v+h*acceleration, self.speed_cap_m_s)
            acceleration = (next_v-v)/h
            p = p+h*v+.5*h*h*acceleration
            v = next_v
        return p, v, []


def install_state_only_prediction(network, belief=None):
    """Replace only predictor-owned model references, never the world object."""
    if not isinstance(network._model, VelocityResponse):
        raise TypeError('Expected the explicit network matched-model predictor')
    network._model = StateOnlyVelocityResponse(**asdict(network._model))
    if belief is not None:
        if not isinstance(belief.model, VelocityResponse):
            raise TypeError('Expected the explicit bounded-belief matched model')
        belief.model = StateOnlyVelocityResponse(**asdict(belief.model))

