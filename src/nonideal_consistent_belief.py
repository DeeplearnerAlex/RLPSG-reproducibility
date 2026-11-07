"""Command-consistent bounded belief with strictly contained packet fusion.

Only delivered ObservationFrame data and reliable known commands are accepted.
There are NO true-state inputs, truth checks or oracle command repairs here.
The first fully valid frame initializes independent position/velocity balls.
Fresh packet balls replace them only if completely contained in the predicted
prior balls, separately for p and v; overlapping but non-contained balls are
rejected rather than moving the uncertainty tube discontinuously.

This deliberately conservative rule may reject almost every fresh velocity
packet after its prior radius contracts. It then relies on the initial bounded
measurement and the EXACT matched VelocityResponse plant/known command history.
It is not robust to arbitrary model mismatch, unknown external forces, unreliable
downlink or unbounded real sensor errors, and is not a high-fidelity claim.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import copy

import numpy as np

from nonideal_observation_network import ObservationFrame
from reduced_velocity_dynamics import VelocityResponse, norm_clip


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


class ConsistentBelief:
    """``update(raw_frame) -> fused frame`` and ``advance(command, dt)``.

    Initialize only after network bootstrap, then call advance with exactly the
    held command/time interval used by the real matched plant. A new frame must
    represent that propagated clock. raw ages/sensed pairs remain unchanged;
    fused valid means reliable model propagation, not receipt of a fresh packet.
    """
    def __init__(self, model=None, numerical_margin=1e-12):
        self.model = VelocityResponse() if model is None else model
        if not isinstance(self.model, VelocityResponse):
            raise TypeError('Expected the explicitly matched VelocityResponse model')
        if not np.isfinite(numerical_margin) or numerical_margin < 0:
            raise ValueError('A finite nonnegative numerical radius margin is required')
        self.numerical_margin = float(numerical_margin)
        self.population = None
        self._p = self._v = self._ep = self._ev = None
        self._time = None
        self._updates = self._advances = 0
        self._totals = dict(position_accepted=0, position_rejected=0, position_invalid=0,
                            velocity_accepted=0, velocity_rejected=0, velocity_invalid=0,
                            contractive_robot_steps=0, nonexpansive_robot_steps=0,
                            raw_stale_observations=0, propagated_seconds=0.)

    @property
    def initialized(self):
        return self._p is not None

    @staticmethod
    def _frame_time(frame):
        timestamps = np.asarray(frame.measurement_timestamps, dtype=float)
        ages = np.asarray(frame.ages, dtype=float)
        usable = np.isfinite(timestamps) & np.isfinite(ages)
        if not usable.any():
            return None
        clocks = timestamps[usable]+ages[usable]
        if not np.isfinite(clocks).all() or clocks.min() < 0 or clocks.max()-clocks.min() > 1e-8:
            raise ValueError('Observation frame has inconsistent measurement/age clocks')
        return float(clocks[0])

    def _validate_frame(self, frame):
        if not isinstance(frame, ObservationFrame):
            raise TypeError('A delivered ObservationFrame is required, not world state')
        p, v = np.asarray(frame.p_hat), np.asarray(frame.v_hat)
        if p.ndim != 2 or p.shape[1] != 2 or not len(p) or v.shape != p.shape:
            raise ValueError('Expected matching nonempty N-by-2 observed states')
        n = len(p)
        for name in ['valid', 'ages', 'measurement_timestamps', 'uncertainty_position', 'uncertainty_velocity']:
            if np.asarray(getattr(frame, name)).shape != (n,):
                raise ValueError(f'Observation frame shape mismatch: {name}')
        if self.population is not None and n != self.population:
            raise ValueError('Belief population changed')
        return p, v, np.asarray(frame.valid, dtype=bool)

    def _statistics(self, raw, stale, last):
        values = dict(raw.stats)
        values.update(raw_stale_ids=np.flatnonzero(stale).astype(int).tolist(),
                      raw_valid_fraction=float(np.mean(~stale)),
                      consistent_belief=dict(initialized=self.initialized,
                          population=self.population, model_time_s=self._time,
                          updates=self._updates, advances=self._advances,
                          **self._totals, **last,
                          validity='Initialized bounded belief under reliable known input and exact matched model; raw stale does not invalidate propagation',
                          fusion_rule='Separate complete-ball containment, never intersection center reset or true-state clipping',
                          model_dependency='May reject most fresh velocity packets and rely on initial measurement plus the matched discrete plant; no unknown-dynamics robustness claim'))
        return values

    def update(self, frame):
        p, v, raw_valid = self._validate_frame(frame)
        n = len(p)
        ep, ev = np.asarray(frame.uncertainty_position, dtype=float), np.asarray(frame.uncertainty_velocity, dtype=float)
        usable_p = raw_valid & np.isfinite(p).all(axis=1) & np.isfinite(ep) & (ep >= 0)
        usable_v = raw_valid & np.isfinite(v).all(axis=1) & np.isfinite(ev) & (ev >= 0)
        frame_time = self._frame_time(frame)
        self._updates += 1
        last = dict(last_position_accepted=0, last_position_rejected=0,
                    last_velocity_accepted=0, last_velocity_rejected=0)
        self._totals['raw_stale_observations'] += int((~raw_valid).sum())
        if not self.initialized:
            if not (usable_p.all() and usable_v.all() and frame_time is not None):
                # No hidden bootstrap: the caller receives the original
                # unavailable observations, not dummy zero or true positions.
                return replace(frame, stats=self._statistics(frame, ~raw_valid, last))
            self.population = n
            self._p = p.astype(float, copy=True)
            # Projection onto the known speed ball is nonexpansive relative to
            # a true state obeying that declared plant speed bound.
            self._v = norm_clip(v, self.model.speed_cap_m_s)
            self._ep, self._ev = ep.copy(), ev.copy()
            self._time = frame_time
            last['initialized_from_full_valid_observation'] = True
        else:
            if frame_time is None or abs(frame_time-self._time) > 1e-8:
                raise ValueError('Frame clock disagrees with reliable-command belief propagation')
            distance_p = np.linalg.norm(p-self._p, axis=1)
            distance_v = np.linalg.norm(v-self._v, axis=1)
            usable_v &= np.linalg.norm(v, axis=1) <= self.model.speed_cap_m_s+1e-9
            # Strict containment: there is deliberately NO positive acceptance
            # tolerance which could move a posterior ball outside its prior.
            accept_p = usable_p & (distance_p+ep <= self._ep)
            accept_v = usable_v & (distance_v+ev <= self._ev)
            self._p[accept_p] = p[accept_p]; self._ep[accept_p] = ep[accept_p]
            self._v[accept_v] = v[accept_v]; self._ev[accept_v] = ev[accept_v]
            for label, okay, accepted in [('position', usable_p, accept_p), ('velocity', usable_v, accept_v)]:
                yes, no, invalid = int(accepted.sum()), int((okay & ~accepted).sum()), int((~okay).sum())
                self._totals[label+'_accepted'] += yes
                self._totals[label+'_rejected'] += no
                self._totals[label+'_invalid'] += invalid
                last['last_'+label+'_accepted'] = yes
                last['last_'+label+'_rejected'] = no
        return replace(frame, p_hat=_readonly(self._p), v_hat=_readonly(self._v),
                       uncertainty_position=_readonly(self._ep), uncertainty_velocity=_readonly(self._ev),
                       valid=_readonly(np.ones(n, dtype=bool)),
                       stats=self._statistics(frame, ~raw_valid, last))

    def advance(self, command, dt):
        if not self.initialized:
            raise ValueError('Cannot propagate a belief before a complete bounded observation bootstrap')
        command, dt = np.asarray(command, dtype=float), float(dt)
        if command.shape != self._p.shape or not np.isfinite(command).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Expected a finite held command and positive propagation duration')
        c = norm_clip(command, self.model.speed_cap_m_s)
        steps = int(np.ceil(dt/self.model.integration_dt_s)); h = dt/steps
        if h > self.model.tau_s:
            raise ValueError('Nonexpansive/contractive propagation requires h<=tau')
        robust_unsaturated = (np.linalg.norm(c-self._v, axis=1)+self._ev <=
                              self.model.tau_s*self.model.acceleration_cap_m_s2)
        # This stronger whole-ball check also suffices for both the actual and
        # estimated speeds to obey the common speed cap. Outside it, retain the
        # global nonexpansive bound rather than assuming exponential decay.
        whole_speed_ball = np.linalg.norm(self._v, axis=1)+self._ev <= self.model.speed_cap_m_s
        contractive = robust_unsaturated & whole_speed_ball
        q = (1-h/self.model.tau_s)**steps
        A = (self.model.tau_s-h/2)*(1-q)
        old_ev = self._ev.copy()
        # Pass the original held command: VelocityResponse clips it exactly
        # once, just as the real matched world does (also for ideal bit replay).
        self._p, self._v, _ = self.model.advance(self._p, self._v, command, dt)
        self._ep += np.where(contractive, A, dt)*old_ev
        self._ev = np.where(contractive, q*old_ev, old_ev)
        # Keep a tiny numerical guard for nonzero uncertainty, while the exact
        # zero-noise ideal path remains exactly zero and bitwise replayable.
        uncertain = (self._ep > 0) | (self._ev > 0)
        self._ep[uncertain] += self.numerical_margin
        self._ev[uncertain] += self.numerical_margin
        self._time += dt
        self._advances += 1
        self._totals['propagated_seconds'] += dt
        self._totals['contractive_robot_steps'] += int(contractive.sum())
        self._totals['nonexpansive_robot_steps'] += int((~contractive).sum())
        return dict(steps=steps, integration_h_s=h, q=q, A_s=A,
                    contractive_robots=int(contractive.sum()), nonexpansive_robots=int((~contractive).sum()),
                    model_time_s=self._time)

    def state_dict(self):
        return dict(schema_version=1, model=asdict(self.model), numerical_margin=self.numerical_margin,
                    population=self.population, time_s=self._time, updates=self._updates, advances=self._advances,
                    p=None if not self.initialized else self._p.tolist(),
                    v=None if not self.initialized else self._v.tolist(),
                    ep=None if not self.initialized else self._ep.tolist(),
                    ev=None if not self.initialized else self._ev.tolist(), totals=copy.deepcopy(self._totals))

    def load_state_dict(self, state):
        if (state['schema_version'] != 1 or state['model'] != asdict(self.model)
                or state['numerical_margin'] != self.numerical_margin):
            raise ValueError('Consistent-belief checkpoint model/configuration changed')
        n = state['population']
        if n is None:
            if any(state[name] is not None for name in ['p', 'v', 'ep', 'ev', 'time_s']):
                raise ValueError('An uninitialized belief cannot contain invented state')
            self.population = None; self._p = self._v = self._ep = self._ev = self._time = None
        else:
            if not isinstance(n, int) or n < 1:
                raise ValueError('Invalid belief population')
            p, v, ep, ev = [np.asarray(state[name], dtype=float) for name in ['p', 'v', 'ep', 'ev']]
            if (p.shape != (n, 2) or v.shape != p.shape or ep.shape != (n,) or ev.shape != (n,)
                    or not all(np.isfinite(value).all() for value in [p, v, ep, ev])
                    or np.any(ep < 0) or np.any(ev < 0)
                    or np.linalg.norm(v, axis=1).max() > self.model.speed_cap_m_s+1e-9
                    or not np.isfinite(state['time_s']) or state['time_s'] < 0):
                raise ValueError('Invalid bounded belief checkpoint arrays/time')
            self.population, self._time = n, float(state['time_s'])
            self._p, self._v, self._ep, self._ev = [value.copy() for value in [p, v, ep, ev]]
        self._updates, self._advances = int(state['updates']), int(state['advances'])
        self._totals = copy.deepcopy(state['totals'])

