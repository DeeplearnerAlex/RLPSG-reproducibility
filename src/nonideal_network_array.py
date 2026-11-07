"""Exact delivered-pair assembly acceleration; sensor/network rules unchanged.

Only the private delivered-cache view is replaced. Measurements, random draws,
packet delivery, command replay, uncertainty bounds and checkpoints are inherited
unchanged. No truth graph is constructed here.
"""
from __future__ import annotations

import numpy as np

from nonideal_observation_network import CombinedObservationNetwork, ObservationFrame, _readonly


def delivered_pairs(network, valid):
    """Same newest-time/smaller-observer/first-tie selection as the old dict."""
    lengths = np.fromiter(map(len, network._cache_neighbor_ids), dtype=np.int64,
                          count=network.population)
    selected = np.flatnonzero(valid & (lengths > 0))
    if not len(selected):
        return (np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=float),
                np.empty(0, dtype=float), np.empty(0, dtype=np.int64))
    observers = np.repeat(selected, lengths[selected])
    neighbors = np.concatenate([network._cache_neighbor_ids[i] for i in selected])
    relative = np.concatenate([network._cache_neighbor_rel[i] for i in selected])
    timestamps = network._cache_t[observers]
    first, second = np.minimum(observers, neighbors), np.maximum(observers, neighbors)
    # An explicit ordinal preserves the first row on an exact restored-cache tie.
    order = np.lexsort((np.arange(len(observers)), observers, -timestamps, second, first))
    keep = np.r_[True, (first[order[1:]] != first[order[:-1]]) |
                  (second[order[1:]] != second[order[:-1]])]
    chosen = order[keep]
    values = relative[chosen].copy()
    reverse = observers[chosen] != first[chosen]
    values[reverse] = -values[reverse]
    return (np.column_stack((first[chosen], second[chosen])).astype(np.int64),
            values, timestamps[chosen], observers[chosen].astype(np.int64))


class ArrayObservationNetwork(CombinedObservationNetwork):
    """Public interface/checkpoint schema identical to the reference network."""
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
        pairs, relative, pair_t, observers = delivered_pairs(self, valid)
        pair_age = time_s-pair_t
        finite_ages = age[cached]
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

