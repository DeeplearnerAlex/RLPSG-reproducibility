"""Observation-only terminal allocation for the nonideal physical simulator.

Zero-target estimates use the existing persistent cleanup router.  Positive-
target estimates may continue receiving ordinary PPO proposals.  The caller
executes goal bins through the unchanged motion planner/plant, NOT a macro
teleport.  A previous command is never interpreted as an observed arrival.

This module has no true-state, world, sensor simulator, or motion-filter inputs.
Its integer capacity accounting concerns proposed goal assignments conditional
on the supplied estimates; physical/observed counts need not be monotone.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from terminal_positive_rebalance import TerminalPositiveRebalanceAllocator


def restrict_positive_transition_matrix(matrix, target):
    """Condition positive-source columns on positive destinations.

    Zero-source columns are retained, although cleanup replaces their samples.
    Empty columns receive self probability one.  No nonlocal edge is created.
    Nonzero input columns must already be stochastic within 1e-5; all-zero
    columns are allowed specifically for the explicit self-loop fallback.
    Apply BEFORE relative-position adjustment and target-bin sampling.
    """
    phi = np.asarray(target, dtype=float)
    weights = np.array(matrix, dtype=float, copy=True)
    if (phi.ndim != 1 or not len(phi) or not np.isfinite(phi).all()
            or (phi < 0).any() or phi.sum() <= 0):
        raise ValueError('Invalid target density')
    if (weights.shape != (len(phi), len(phi)) or not np.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError('Invalid destination-by-source transition matrix')
    totals = weights.sum(axis=0)
    if ((totals != 0) & (np.abs(totals - 1.) > 1e-5)).any():
        raise ValueError('Nonzero input columns must be stochastic')
    positive = phi > 0
    weights[np.ix_(~positive, positive)] = 0.
    totals = weights.sum(axis=0)
    empty = np.flatnonzero(totals == 0)
    weights[empty, empty] = 1.
    weights /= weights.sum(axis=0, keepdims=True)
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or np.max(np.abs(weights.sum(axis=0) - 1.)) > 1e-12):
        raise RuntimeError('Restricted transition matrix is not stochastic')
    return weights


class ObservedTerminalPositiveAllocator(TerminalPositiveRebalanceAllocator):
    """Continuous-observation adapter around the audited macro admission rule.

    allocate(points, positive_proposals, velocity=None, counts=None,
             allocation_index=None) -> (goal_bins, report)
    observe(points, velocity=None, counts=None, allocation_index=None) -> report

    ``points`` and optional ``velocity`` are delivered estimates.  ``counts``,
    when supplied, must equal their bin counts.  Fixed stable identity ordering
    is required, but motion between calls may be incomplete, overshoot a goal,
    or cross an uncommanded bin.  Those observations cause bookkeeping/replans,
    never a true-state correction or forced declaration of arrival.

    As in Experiment I, source/receiver gates use the PRE-observation float32
    density, cleanup reservations have priority, and positive proposals share
    remaining integer quota in ID order.  Positive proposal commitments last
    for one allocation phase; only zero-cleanup reservations persist.
    """

    VARIANT = 'zero_cleanup_positive_ppo_observed'

    def __init__(self, target, population, domain_side=800., grid=None):
        super().__init__(target, population, domain_side=domain_side, grid=grid)
        self._last_observed_bins = None
        self._last_goal_bins = None
        self._adapter_events = []
        self._events_this_observation = []
        self._context_index = None

    def _observed_bins(self, points, velocity=None, counts=None):
        p = np.asarray(points, dtype=float)
        if (p.shape != (self.population, 2) or not np.isfinite(p).all()
                or ((p < 0) | (p >= self.domain_side)).any()):
            raise ValueError('Expected finite in-domain N-by-2 delivered position estimates')
        if velocity is not None:
            v = np.asarray(velocity, dtype=float)
            if v.shape != p.shape or not np.isfinite(v).all():
                raise ValueError('Expected finite N-by-2 delivered velocity estimates')
        xy = np.floor(p / (self.domain_side / self.grid)).astype(np.int64)
        bins = xy[:, 0] + self.grid * xy[:, 1]
        observed = np.bincount(bins, minlength=len(self.target))
        if counts is not None:
            supplied = np.asarray(counts)
            if (supplied.shape != observed.shape or not np.isfinite(supplied).all()
                    or not np.array_equal(supplied, observed)):
                raise ValueError('Supplied counts disagree with delivered position estimates')
        return bins

    def _actual(self, locations):
        # The macro parent's equality-to-previous-goals check MUST NOT be used
        # for continuous motion: a target command does not establish arrival.
        return self._locations(locations, 'observed_bins')

    def _sync_positive_placeholders(self, bins):
        if self._cleanup._records is None:
            return
        for record, observed in zip(self._cleanup._records, bins):
            if record['status'] not in ('frozen', 'delivered'):
                # Inflight observations entering any positive bin are left to
                # the existing router's delivered/unexpected-delivery logic.
                continue
            observed = int(observed)
            if self.target[observed] > 0:
                record['locked_bin'] = observed
                record['next_hop'] = observed
                if record['status'] == 'delivered':
                    record.update(final_bin=observed, route=[observed], route_index=0)
                continue
            old = record['locked_bin']
            event = dict(type='observed_positive_reentered_zero',
                         allocation_index=self._context_index, id=record['id'],
                         previous_positive_bin=old, observed_zero_bin=observed,
                         reason='Observed motion/estimation deviation, not a commanded positive-to-zero move')
            self._adapter_events.append(event)
            self._events_this_observation.append(copy.deepcopy(event))
            # Release the old placeholder lock.  Re-entered background robots
            # are routed from the OBSERVED bin, not pulled to their old target.
            record.update(kind='cleanup', status='pending', locked_bin=None,
                          route=[], route_index=0, final_bin=None, reserved_bin=None,
                          next_hop=observed, last_observed_bin=observed,
                          blocked_reason=None)

    def _observation_report(self, raw, bins, observation_only):
        report = copy.deepcopy(raw)
        report['observed_counts'] = report.pop('counts_before')
        report['observed_hd'] = report.pop('hd_before')
        report['observed_zero_target_population'] = report.pop('zero_target_population_before')
        projected = report.pop('counts_after')
        projected_hd = report.pop('hd_after')
        projected_zero = report.pop('zero_target_population_after')
        report['projected_counts_if_goals_reached'] = None if observation_only else projected
        report['projected_hd_if_goals_reached'] = None if observation_only else projected_hd
        report['projected_zero_if_goals_reached'] = None if observation_only else projected_zero
        report['observed_bins'] = bins.tolist()
        report['adapter_events_this_observation'] = copy.deepcopy(self._events_this_observation)
        report['goal_arrival_assumed'] = False
        report['scope'] = ('Observation-only goal allocation; projected assignment counts are not '
                           'measured physical arrivals or safety/convergence guarantees')
        report['positive_commitment'] = ('One allocation phase; sampled anew from the next '
                                          'delivered estimates, even after a phase deadline')
        report['physical_or_observed_zero_monotonicity_claimed'] = False
        self.last_report = report
        self._last_observed_bins = bins.copy()
        return report

    def allocate(self, points, positive_proposals, velocity=None, counts=None,
                 allocation_index=None):
        bins = self._observed_bins(points, velocity, counts)
        proposals = self._locations(positive_proposals, 'positive_proposals')
        self._context_index = self._index(allocation_index)
        self._events_this_observation = []
        goals, raw = super().allocate(bins, proposals, self._context_index)
        self._last_goal_bins = goals.copy()
        self._expected_next = None  # Never expose the macro arrival assumption.
        return goals, self._observation_report(raw, bins, observation_only=False)

    def observe(self, points, velocity=None, counts=None, allocation_index=None):
        bins = self._observed_bins(points, velocity, counts)
        self._context_index = self._index(allocation_index)
        self._events_this_observation = []
        raw = super().observe(bins, self._context_index)
        self._expected_next = None
        return self._observation_report(raw, bins, observation_only=True)

    def state_dict(self):
        """JSON continuation: measured bins and commanded goals stay separate."""
        return copy.deepcopy(dict(
            version=1, variant=self.VARIANT, target=self.target.tolist(),
            population=self.population, grid=self.grid, domain_side=self.domain_side,
            quota=self.quota.tolist(), calls=self._calls, observations=self._observations,
            cleanup_routing_active=self._cleanup_active,
            routing_engine_state=self._cleanup.state_dict(),
            last_observed_bins=None if self._last_observed_bins is None else self._last_observed_bins.tolist(),
            last_goal_bins=None if self._last_goal_bins is None else self._last_goal_bins.tolist(),
            adapter_events=self._adapter_events, last_report=self.last_report,
            continuation_contract='Next input is a fresh delivered estimate, not the last goal bins'))

    def load_state_dict(self, state):
        try:
            value = json.loads(json.dumps(state, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError('State must be finite JSON data') from error
        for key, expected in [('version', 1), ('variant', self.VARIANT),
                              ('target', self.target.tolist()), ('population', self.population),
                              ('grid', self.grid), ('domain_side', self.domain_side),
                              ('quota', self.quota.tolist())]:
            if value.get(key) != expected:
                raise ValueError(f'State configuration mismatch: {key}')
        for key in ('calls', 'observations'):
            if (not isinstance(value.get(key), int) or isinstance(value[key], bool)
                    or value[key] < 0):
                raise ValueError(f'Invalid {key}')
        if not isinstance(value.get('cleanup_routing_active'), bool):
            raise ValueError('Invalid routing-active state')
        observed = value.get('last_observed_bins')
        observed = None if observed is None else self._locations(observed, 'last_observed_bins')
        goals = value.get('last_goal_bins')
        goals = None if goals is None else self._locations(goals, 'last_goal_bins')
        if (value['calls'] and goals is None) or ((value['calls'] or value['observations']) and observed is None):
            raise ValueError('Observed/executed calls require separate observation and goal history')
        report = value.get('last_report')
        events = value.get('adapter_events')
        if not isinstance(report, dict) or not isinstance(events, list):
            raise ValueError('Invalid report/event history')
        if observed is not None:
            counts = np.bincount(observed, minlength=len(self.target))
            if not np.array_equal(report.get('observed_counts'), counts):
                raise ValueError('Report disagrees with last observed bins')
            if report.get('goal_arrival_assumed') is not False:
                raise ValueError('State must not assume command arrival')
            if not report.get('observation_only'):
                if goals is None or not np.array_equal(
                        report.get('projected_counts_if_goals_reached'),
                        np.bincount(goals, minlength=len(self.target))):
                    raise ValueError('Projected counts disagree with last commanded goals')
        # The dormant router can contain an older all-positive observation;
        # its state is not asserted to be the last physical/estimated endpoint.
        self._cleanup.load_state_dict(value['routing_engine_state'])
        self._cleanup_active = value['cleanup_routing_active']
        self._calls, self._observations = value['calls'], value['observations']
        self._last_observed_bins, self._last_goal_bins = observed, goals
        self._adapter_events = events
        self.last_report = report
        self._events_this_observation = []
        self._expected_next = None
        self._context_index = None
        return self

