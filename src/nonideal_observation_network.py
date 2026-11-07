"""Measurement/uplink simulator with no truth output to the controller.

The world supplies true p/v ONLY to ``observe`` for sensor generation. Returned
frames are reconstructed solely from delivered measurement packets and recorded
reliable-downlink commands. Delay/loss apply to the UPLINK, not both links.

Default errors are preset sensitivity parameters, NOT calibrated sensor specs.
Bias, clipped Gauss--Markov drift and disk jitter have deterministic norm bounds;
they are not unbounded Gaussian localization errors. Under the EXACT matched
VelocityResponse model, recorded commands and aligned integration, the capped
velocity update is nonexpansive, so replay error obeys e_v<=initial e_v and
e_p<=initial e_p+age*initial e_v. This conditional numerical-model bound is not
a guarantee for heterogeneous/unmodelled actuators or real vehicles.

Circle obstacles have shape (M,3): center x, center y, radius. Finite-range
relative measurements are timestamped and become visible only when the owning
robot's uplink packet arrives; they are NOT a current true-neighbor graph.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import copy
import math

import numpy as np
from scipy.spatial import cKDTree

from reduced_velocity_dynamics import VelocityResponse, norm_clip


@dataclass(frozen=True)
class ObservationConfig:
    measurement_period_s: float = .1
    delay_choices_s: tuple[float, ...] = (.1, .2)
    delay_jitter_s: float = .02
    packet_loss_probability: float = .05
    position_bias_bound_m: float = .02
    drift_bound_m: float = .03
    drift_stationary_std_m: float = .01
    drift_time_constant_s: float = 5.
    position_jitter_bound_m: float = .02
    velocity_noise_bound_m_s: float = .02
    sensing_range_m: float = 3.
    relative_noise_bound_m: float = .02
    sensing_miss_probability: float = .05
    stale_after_s: float = .6
    predictor: str = 'known_command'
    tau_s: float = .5
    speed_cap_m_s: float = 1.
    acceleration_cap_m_s2: float = 1.
    integration_dt_s: float = .01
    command_history_s: float = 5.
    numerical_prediction_margin: float = 1e-8

    def __post_init__(self):
        nonnegative = ['delay_jitter_s', 'position_bias_bound_m', 'drift_bound_m',
                       'drift_stationary_std_m', 'position_jitter_bound_m',
                       'velocity_noise_bound_m_s', 'sensing_range_m', 'relative_noise_bound_m',
                       'numerical_prediction_margin']
        positive = ['measurement_period_s', 'drift_time_constant_s', 'stale_after_s',
                    'tau_s', 'speed_cap_m_s', 'acceleration_cap_m_s2', 'integration_dt_s', 'command_history_s']
        for name in nonnegative+positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f'Invalid observation configuration: {name}')
        for name in ['packet_loss_probability', 'sensing_miss_probability']:
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'Invalid probability: {name}')
        if not self.delay_choices_s or any(not np.isfinite(t) or t < 0 for t in self.delay_choices_s):
            raise ValueError('Delay choices must be finite and nonnegative')
        object.__setattr__(self, 'delay_choices_s', tuple(float(t) for t in self.delay_choices_s))
        if self.predictor not in ['known_command', 'constant_velocity']:
            raise ValueError('Unknown observation predictor')
        if self.command_history_s < self.stale_after_s+max(self.delay_choices_s)+self.delay_jitter_s:
            raise ValueError('Command history must cover valid aged packets and maximum transmission delay')
        VelocityResponse(self.tau_s, self.speed_cap_m_s, self.acceleration_cap_m_s2, self.integration_dt_s)

    @classmethod
    def ideal(cls, **changes):
        values = dict(delay_choices_s=(0.,), delay_jitter_s=0., packet_loss_probability=0.,
                      position_bias_bound_m=0., drift_bound_m=0., drift_stationary_std_m=0.,
                      position_jitter_bound_m=0., velocity_noise_bound_m_s=0.,
                      relative_noise_bound_m=0., sensing_miss_probability=0., numerical_prediction_margin=0.)
        values.update(changes)
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if 'delay_choices_s' in values:
            values['delay_choices_s'] = tuple(values['delay_choices_s'])
        return cls(**values)

    @property
    def measurement_position_bound_m(self):
        return self.position_bias_bound_m+self.drift_bound_m+self.position_jitter_bound_m


@dataclass(frozen=True)
class ObservationFrame:
    p_hat: np.ndarray
    v_hat: np.ndarray
    ages: np.ndarray
    valid: np.ndarray
    uncertainty_position: np.ndarray
    uncertainty_velocity: np.ndarray
    measurement_timestamps: np.ndarray
    cached_positions: np.ndarray
    cached_velocities: np.ndarray
    sensed_pairs: np.ndarray
    sensed_relative_positions: np.ndarray
    sensed_pair_timestamps: np.ndarray
    sensed_pair_ages: np.ndarray
    sensed_pair_observers: np.ndarray
    sensed_pair_uncertainty_position: np.ndarray
    stats: dict


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _disk_noise(rng, number, bound):
    # Uniform disk, with a known Euclidean (not per-axis) error bound.
    angle = rng.uniform(0., 2*np.pi, number)
    radius = bound*np.sqrt(rng.random(number))
    return radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)])


def _circles(obstacles):
    result = np.empty((0, 3), dtype=float) if obstacles is None else np.asarray(obstacles, dtype=float)
    if result.size == 0:
        return np.empty((0, 3), dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all() or np.any(result[:, 2] < 0):
        raise ValueError('Obstacles must be finite circle rows (x,y,radius)')
    return result


def _occluded(points, pairs, obstacles):
    hidden = np.zeros(len(pairs), dtype=bool)
    if not len(pairs):
        return hidden
    start, end = points[pairs[:, 0]], points[pairs[:, 1]]
    segment = end-start
    denominator = np.maximum(np.sum(segment*segment, axis=1), 1e-30)
    for x, y, radius in obstacles:
        center = np.array([x, y])
        parameter = np.clip(np.sum((center-start)*segment, axis=1)/denominator, 0., 1.)
        closest = start+parameter[:, None]*segment
        hidden |= np.sum((closest-center)**2, axis=1) <= radius*radius+1e-15
    return hidden


class CombinedObservationNetwork:
    """Reproducible bounded sensors, delayed/lossy uplink and known-input replay.

    Call ``observe(t,p_true,v_true,circles)`` before the control calculation, then
    ``record_command(t,executed_held_command)`` BEFORE advancing the real plant.
    Commands are reliable-downlink inputs known to the controller, never true
    state. Unreceived startup identities remain NaN/invalid, not truth-filled.
    """
    def __init__(self, population, seed, config=None):
        if not isinstance(population, (int, np.integer)) or population < 1:
            raise ValueError('Population must be a positive integer')
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        self.population, self.seed = int(population), int(seed)
        self.config = (ObservationConfig() if config is None else
                       ObservationConfig.from_dict(config) if isinstance(config, dict) else config)
        if not isinstance(self.config, ObservationConfig):
            raise TypeError('Expected ObservationConfig or its dictionary')
        streams = np.random.SeedSequence([self.seed, 110503]).spawn(3)
        self._measurement_rng, self._network_rng, self._sensing_rng = [np.random.default_rng(s) for s in streams]
        self._bias = _disk_noise(self._measurement_rng, self.population, self.config.position_bias_bound_m)
        self._drift = np.zeros((self.population, 2), dtype=float)
        self._cache_p = np.zeros((self.population, 2), dtype=float)
        self._cache_v = np.zeros_like(self._cache_p)
        self._cache_t = np.full(self.population, np.nan)
        self._cache_neighbor_ids = [np.empty(0, dtype=np.int64) for _ in range(self.population)]
        self._cache_neighbor_rel = [np.empty((0, 2)) for _ in range(self.population)]
        self._pending = []
        self._command_history = []
        self._last_observe = self._last_measurement = self._next_measurement = None
        self._totals = dict(observe_calls=0, measurement_batches=0, packets_generated=0,
                            packets_lost=0, packets_delivered=0, older_packets_ignored=0,
                            sensing_candidates=0, sensing_occluded=0, sensing_missed=0,
                            missed_measurement_slots=0)
        c = self.config
        self._model = VelocityResponse(c.tau_s, c.speed_cap_m_s, c.acceleration_cap_m_s2, c.integration_dt_s)

    def record_command(self, time_s, commands):
        time_s = float(time_s)
        commands = np.asarray(commands, dtype=float)
        if not np.isfinite(time_s) or time_s < 0 or commands.shape != (self.population, 2) or not np.isfinite(commands).all():
            raise ValueError('A finite timestamp and one finite command per identity are required')
        if self._command_history:
            last_time, last = self._command_history[-1]
            if time_s < last_time:
                raise ValueError('Recorded command times must be chronological')
            if time_s == last_time:
                if not np.array_equal(commands, last):
                    raise ValueError('Cannot overwrite a previously recorded reliable-downlink command')
                return
        self._command_history.append((time_s, commands.copy()))

    def _sense(self, points, obstacles):
        c = self.config
        rows = [(np.empty(0, dtype=np.int64), np.empty((0, 2))) for _ in range(self.population)]
        if c.sensing_range_m == 0 or self.population < 2:
            return rows
        pairs = cKDTree(points).query_pairs(c.sensing_range_m, output_type='ndarray')
        if not len(pairs):
            return rows
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        hidden = _occluded(points, pairs, obstacles)
        self._totals['sensing_candidates'] += 2*len(pairs)
        self._totals['sensing_occluded'] += 2*int(hidden.sum())
        pairs = pairs[~hidden]
        if not len(pairs):
            return rows
        observers = np.r_[pairs[:, 0], pairs[:, 1]]
        neighbors = np.r_[pairs[:, 1], pairs[:, 0]]
        order = np.lexsort((neighbors, observers))
        observers, neighbors = observers[order], neighbors[order]
        keep = self._sensing_rng.random(len(observers)) >= c.sensing_miss_probability
        self._totals['sensing_missed'] += int((~keep).sum())
        observers, neighbors = observers[keep], neighbors[keep]
        relative = points[neighbors]-points[observers]+_disk_noise(self._sensing_rng, len(observers), c.relative_noise_bound_m)
        starts = np.r_[0, np.flatnonzero(np.diff(observers))+1] if len(observers) else []
        ends = np.r_[starts[1:], len(observers)] if len(observers) else []
        for start, end in zip(starts, ends):
            rows[int(observers[start])] = (neighbors[start:end].astype(np.int64), relative[start:end].copy())
        return rows

    def _measure(self, time_s, points, velocity, obstacles):
        c = self.config
        if self._last_measurement is not None:
            dt = time_s-self._last_measurement
            rho = math.exp(-dt/c.drift_time_constant_s)
            innovation = self._measurement_rng.normal(0., c.drift_stationary_std_m*math.sqrt(max(0., 1-rho*rho)), self._drift.shape)
            self._drift = norm_clip(rho*self._drift+innovation, c.drift_bound_m)
        measured_p = points+self._bias+self._drift+_disk_noise(self._measurement_rng, self.population, c.position_jitter_bound_m)
        measured_v = norm_clip(velocity+_disk_noise(self._measurement_rng, self.population, c.velocity_noise_bound_m_s), c.speed_cap_m_s)
        neighbor_rows = self._sense(points, obstacles)
        retained = self._network_rng.random(self.population) >= c.packet_loss_probability
        delay = self._network_rng.choice(np.asarray(c.delay_choices_s), self.population)
        delay += self._network_rng.uniform(-c.delay_jitter_s, c.delay_jitter_s, self.population)
        ids = np.flatnonzero(retained)
        if len(ids):
            self._pending.append(dict(timestamp=time_s, ids=ids, arrival=time_s+np.maximum(delay[ids], 0.),
                p=measured_p[ids].copy(), v=measured_v[ids].copy(), neighbors=[neighbor_rows[i] for i in ids]))
        self._totals['measurement_batches'] += 1
        self._totals['packets_generated'] += self.population
        self._totals['packets_lost'] += int((~retained).sum())
        self._last_measurement = time_s

    def _deliver(self, time_s):
        pending = []
        for batch in self._pending:
            ready = batch['arrival'] <= time_s+1e-12
            positions = np.flatnonzero(ready)
            ids = batch['ids'][ready]
            self._totals['packets_delivered'] += len(ids)
            newer = np.isnan(self._cache_t[ids]) | (batch['timestamp'] > self._cache_t[ids])
            self._totals['older_packets_ignored'] += int((~newer).sum())
            selected = positions[newer]; updated = ids[newer]
            self._cache_p[updated] = batch['p'][selected]
            self._cache_v[updated] = batch['v'][selected]
            self._cache_t[updated] = batch['timestamp']
            for row, identifier in zip(selected, updated):
                neighbors, relative = batch['neighbors'][int(row)]
                self._cache_neighbor_ids[int(identifier)] = neighbors.copy()
                self._cache_neighbor_rel[int(identifier)] = relative.copy()
            remaining = np.flatnonzero(~ready)
            if len(remaining):
                pending.append(dict(timestamp=batch['timestamp'], ids=batch['ids'][remaining],
                    arrival=batch['arrival'][remaining], p=batch['p'][remaining], v=batch['v'][remaining],
                    neighbors=[batch['neighbors'][int(i)] for i in remaining]))
        self._pending = pending

    def _replay(self, ids, source_time, target_time):
        p, v = self._cache_p[ids].copy(), self._cache_v[ids].copy()
        if target_time-source_time <= 1e-12:
            return p, v, True
        times = [item[0] for item in self._command_history]
        index = bisect_right(times, source_time+1e-10)-1
        if index < 0:
            return p+(target_time-source_time)*v, v, False
        cursor = source_time
        while cursor < target_time-1e-12:
            while index+1 < len(times) and times[index+1] <= cursor+1e-10:
                index += 1
            end = min(target_time, times[index+1] if index+1 < len(times) else target_time)
            duration = end-cursor
            if duration <= 1e-12:
                cursor = end
                continue
            # Same held-input/substep boundaries as the actual declared plant;
            # avoid ceil(0.10000000000003/.01) spuriously creating 11 substeps.
            ticks = round(duration/self.config.integration_dt_s)
            if ticks > 0 and abs(duration-ticks*self.config.integration_dt_s) <= 1e-8:
                duration = ticks*self.config.integration_dt_s
            p, v, _ = self._model.advance(p, v, self._command_history[index][1][ids], duration)
            cursor = end
        return p, v, True

    def _frame(self, time_s):
        c = self.config
        cached = np.isfinite(self._cache_t)
        age = np.where(cached, time_s-self._cache_t, np.inf)
        p = np.full_like(self._cache_p, np.nan); v = np.full_like(self._cache_v, np.nan)
        available = np.zeros(self.population, dtype=bool)
        if c.predictor == 'known_command':
            for source_time in np.unique(self._cache_t[cached]):
                ids = np.flatnonzero(self._cache_t == source_time)
                p[ids], v[ids], okay = self._replay(ids, float(source_time), time_s)
                available[ids] = okay
        else:
            p[cached] = self._cache_p[cached]+age[cached, None]*self._cache_v[cached]
            v[cached] = self._cache_v[cached]
            available[cached] = True
        ep = np.full(self.population, np.inf); ev = np.full(self.population, np.inf)
        ep[cached] = c.measurement_position_bound_m+age[cached]*c.velocity_noise_bound_m_s
        ev[cached] = c.velocity_noise_bound_m_s
        cv = cached & ((c.predictor == 'constant_velocity') | ~available)
        ep[cv] += .5*c.acceleration_cap_m_s2*age[cv]**2
        ev[cv] += c.acceleration_cap_m_s2*age[cv]
        margin = cached & (age > 1e-12)
        ep[margin] += c.numerical_prediction_margin; ev[margin] += c.numerical_prediction_margin
        valid = cached & available & (age <= c.stale_after_s+1e-12)
        # A delivered local relative observation remains timestamped; unlike
        # p_hat it is NOT silently promoted to the current time or truth graph.
        pair_records = {}
        for observer in np.flatnonzero(valid):
            timestamp = float(self._cache_t[observer])
            for neighbor, relative in zip(self._cache_neighbor_ids[observer], self._cache_neighbor_rel[observer]):
                i, j = sorted((int(observer), int(neighbor)))
                candidate = (timestamp, int(observer), relative if observer == i else -relative)
                old = pair_records.get((i, j))
                if old is None or (timestamp, -int(observer)) > (old[0], -old[1]):
                    pair_records[(i, j)] = candidate
        keys = sorted(pair_records)
        pairs = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        relative = np.asarray([pair_records[k][2] for k in keys], dtype=float).reshape(-1, 2)
        pair_t = np.asarray([pair_records[k][0] for k in keys], dtype=float)
        observers = np.asarray([pair_records[k][1] for k in keys], dtype=np.int64)
        pair_age = time_s-pair_t
        finite_ages = age[cached]
        # Current undelivered visibility/miss counters depend on sensor-world
        # geometry. Keep them in the offline checkpoint, NOT in controller
        # frames, which must expose only delivered perception information.
        stats = dict({key: value for key, value in self._totals.items() if not key.startswith('sensing_')},
            seed=self.seed, bootstrap_complete=bool(cached.all()),
            cached_identities=int(cached.sum()), valid_identities=int(valid.sum()),
            predictor_unavailable_identities=int((cached & ~available).sum()),
            maximum_age_s=float(finite_ages.max()) if len(finite_ages) else None,
            mean_age_s=float(finite_ages.mean()) if len(finite_ages) else None,
            queued_packets=sum(len(item['ids']) for item in self._pending),
            actual_packet_loss_fraction=self._totals['packets_lost']/self._totals['packets_generated'] if self._totals['packets_generated'] else 0.,
            sensed_pairs=len(pairs), uplink_only_delay_loss=True, downlink_reliable=True,
            predictor=c.predictor, uncertainty_contract='Matched capped VelocityResponse, recorded reliable commands, aligned substeps; not calibrated real-world bounds',
            localization_errors='Bounded norm bias + clipped Gauss-Markov drift + bounded disk jitter; not unbounded Gaussian')
        return ObservationFrame(*[_readonly(value) for value in [p, v, age, valid, ep, ev, self._cache_t,
            np.where(cached[:, None], self._cache_p, np.nan), np.where(cached[:, None], self._cache_v, np.nan),
            pairs, relative, pair_t, pair_age, observers, c.relative_noise_bound_m+2*c.speed_cap_m_s*pair_age]], stats=stats)

    def observe(self, time_s, true_p, true_v, obstacles=None):
        time_s = float(time_s)
        points, velocity = np.asarray(true_p, dtype=float), np.asarray(true_v, dtype=float)
        if (not np.isfinite(time_s) or time_s < 0 or points.shape != (self.population, 2)
                or velocity.shape != points.shape or not np.isfinite(points).all() or not np.isfinite(velocity).all()):
            raise ValueError('World sensor input requires finite N-by-2 positions/velocities and nonnegative time')
        if self._last_observe is not None and time_s < self._last_observe:
            raise ValueError('Observation time cannot move backwards')
        if np.linalg.norm(velocity, axis=1).max() > self.config.speed_cap_m_s+1e-8:
            raise ValueError('World velocity violates the declared matched-model speed contract')
        circles = _circles(obstacles)
        self._totals['observe_calls'] += 1
        if self._next_measurement is None or time_s >= self._next_measurement-1e-12:
            if self._next_measurement is not None:
                self._totals['missed_measurement_slots'] += max(0, int(math.floor((time_s-self._next_measurement+1e-12)/self.config.measurement_period_s)))
            self._measure(time_s, points, velocity, circles)
            self._next_measurement = time_s+self.config.measurement_period_s
        self._deliver(time_s)
        self._last_observe = time_s
        frame = self._frame(time_s)
        # Keep one held-input anchor before the bounded history window.
        if len(self._command_history) > 1:
            times = [item[0] for item in self._command_history]
            first = max(0, bisect_right(times, time_s-self.config.command_history_s)-1)
            self._command_history = self._command_history[first:]
        return frame

    def state_dict(self):
        """Offline checkpoint only; pending undelivered packets are not a frame."""
        return dict(schema_version=1, population=self.population, seed=self.seed, config=self.config.to_dict(),
            rng=[copy.deepcopy(rng.bit_generator.state) for rng in [self._measurement_rng, self._network_rng, self._sensing_rng]],
            bias=self._bias.tolist(), drift=self._drift.tolist(), cache_p=self._cache_p.tolist(), cache_v=self._cache_v.tolist(),
            cache_t=[float(t) if np.isfinite(t) else None for t in self._cache_t],
            cache_neighbors=[dict(ids=ids.tolist(), relative=relative.tolist())
                             for ids, relative in zip(self._cache_neighbor_ids, self._cache_neighbor_rel)],
            pending=[dict(timestamp=b['timestamp'], ids=b['ids'].tolist(), arrival=b['arrival'].tolist(),
                p=b['p'].tolist(), v=b['v'].tolist(), neighbors=[dict(ids=i.tolist(), relative=r.tolist()) for i, r in b['neighbors']])
                     for b in self._pending],
            command_history=[dict(timestamp=t, commands=u.tolist()) for t, u in self._command_history],
            last_observe=self._last_observe, last_measurement=self._last_measurement,
            next_measurement=self._next_measurement, totals=dict(self._totals))

    def load_state_dict(self, state):
        if (state['schema_version'] != 1 or state['population'] != self.population or state['seed'] != self.seed
                or ObservationConfig.from_dict(state['config']) != self.config):
            raise ValueError('Observation-network checkpoint configuration changed')
        for name in ['bias', 'drift', 'cache_p', 'cache_v']:
            value = np.asarray(state[name], dtype=float)
            if value.shape != (self.population, 2) or not np.isfinite(value).all():
                raise ValueError('Invalid observation-network cached numeric state')
            setattr(self, '_'+name, value.copy())
        self._cache_t = np.asarray([np.nan if t is None else t for t in state['cache_t']], dtype=float)
        if self._cache_t.shape != (self.population,) or np.isinf(self._cache_t).any():
            raise ValueError('Invalid cached measurement timestamps')
        def neighbors(row):
            ids = np.asarray(row['ids'], dtype=np.int64)
            relative = np.asarray(row['relative'], dtype=float).reshape(-1, 2)
            if (ids.ndim != 1 or relative.shape != (len(ids), 2) or not np.isfinite(relative).all()
                    or np.any((ids < 0) | (ids >= self.population))):
                raise ValueError('Invalid cached relative measurements')
            return ids.copy(), relative.copy()
        rows = [neighbors(row) for row in state['cache_neighbors']]
        if len(rows) != self.population:
            raise ValueError('Invalid cached sensor-row population')
        self._cache_neighbor_ids, self._cache_neighbor_rel = map(list, zip(*rows))
        self._pending = []
        for batch in state['pending']:
            ids = np.asarray(batch['ids'], dtype=np.int64)
            arrival = np.asarray(batch['arrival'], dtype=float)
            p, v = np.asarray(batch['p'], dtype=float), np.asarray(batch['v'], dtype=float)
            if (ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or np.any((ids < 0) | (ids >= self.population))
                    or p.shape != (len(ids), 2) or v.shape != p.shape or arrival.shape != ids.shape
                    or not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(arrival).all()
                    or np.any(arrival < batch['timestamp']) or len(batch['neighbors']) != len(ids)):
                raise ValueError('Invalid pending observation packets')
            self._pending.append(dict(timestamp=float(batch['timestamp']), ids=ids, arrival=arrival, p=p, v=v,
                                      neighbors=[neighbors(row) for row in batch['neighbors']]))
        self._command_history = []
        for item in state['command_history']:
            self.record_command(item['timestamp'], item['commands'])
        for rng, saved in zip([self._measurement_rng, self._network_rng, self._sensing_rng], state['rng']):
            rng.bit_generator.state = copy.deepcopy(saved)
        self._last_observe, self._last_measurement, self._next_measurement = [state[key] for key in
            ['last_observe', 'last_measurement', 'next_measurement']]
        self._totals = dict(state['totals'])

