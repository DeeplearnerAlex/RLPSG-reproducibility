"""Macroscopic terminal cleanup with continuing positive-target PPO moves.

This is a NEW allocation variant, not a physical motion planner.  The caller
owns the HD latch and the unchanged frozen-policy sampler.  After latching,
zero-target identities follow the existing cleanup router, whereas identities
already in a positive-target bin may continue to another positive-target bin.
Cleanup reservations have priority over sampled positive-target arrivals.

The old router's positive ``locked_bin`` fields are implementation placeholders
only: they are synchronized to the identity's current positive bin before each
observation.  They do not freeze positive-target identities in this wrapper.
No old experiment, checkpoint, or source implementation is changed.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from terminal_zero_cleanup import TerminalZeroCleanupAllocator, DIAGONAL_RULE


class TerminalPositiveRebalanceAllocator:
    """Fixed-population, synchronous, one-bin-hop terminal allocation.

    ``allocate(locations, positive_proposals, allocation_index=None)`` returns
    ``(next_locations, report)``.  Inputs are N-vectors of integer bin indices.
    The caller must execute that complete returned macro update before the
    following call.  A proposal is NOT a command: inadmissible positive moves
    hold their current bin and are explicitly reported.

    Proposals retain the source-surplus and destination-deficit tests at the
    original float32 density precision.  Positive destinations additionally
    admit at most their unreserved integer-quota vacancies, with ascending-ID
    admission when too many identities sample the same destination.  Positive
    target bins are not allowed to send identities back into zero-target bins.
    """

    def __init__(self, target, population, domain_side=800., grid=None):
        self._cleanup = TerminalZeroCleanupAllocator(
            target, population, domain_side=domain_side, grid=grid)
        self.target = self._cleanup.target
        self.population = self._cleanup.population
        self.grid = self._cleanup.grid
        self.domain_side = self._cleanup.domain_side
        self.quota = self._cleanup.quota
        self.best_hd = self._cleanup.best_hd
        self._expected_next = None
        self._cleanup_active = False
        self._calls = 0
        self._observations = 0
        self.last_report = {}

    def _locations(self, value, name):
        array = np.asarray(value)
        if (array.shape != (self.population,)
                or not np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.bool_)
                or (array < 0).any() or (array >= len(self.target)).any()):
            raise ValueError(f'{name} must be a valid integer population-by-1 bin vector')
        return array.astype(np.int64, copy=True)

    def _actual(self, locations):
        bins = self._locations(locations, 'locations')
        if self._expected_next is not None and not np.array_equal(bins, self._expected_next):
            raise ValueError('locations do not equal the previous synchronous macro result')
        return bins

    @staticmethod
    def _index(allocation_index):
        if allocation_index is not None:
            if (not isinstance(allocation_index, (int, np.integer))
                    or isinstance(allocation_index, (bool, np.bool_))
                    or allocation_index < 0):
                raise ValueError('allocation_index must be a nonnegative integer or None')
            return int(allocation_index)
        return None

    def _sync_positive_placeholders(self, bins):
        if self._cleanup._records is None:
            return
        for record, actual in zip(self._cleanup._records, bins):
            if record['status'] not in ('frozen', 'delivered'):
                continue
            actual = int(actual)
            if self.target[actual] <= 0:
                raise ValueError('A delivered/positive identity cannot re-enter a zero-target bin')
            record['locked_bin'] = actual
            record['next_hop'] = actual
            if record['status'] == 'delivered':
                record['final_bin'] = actual
                record['route'] = [actual]
                record['route_index'] = 0

    def _cleanup_plan(self, bins, allocation_index):
        counts = np.bincount(bins, minlength=len(self.target))
        zero = self.target[bins] == 0
        cleanup_goals = bins.copy()
        reserved = np.zeros(len(self.target), dtype=np.int64)
        destinations = np.full(self.population, -1, dtype=np.int64)
        blocked = []
        routes = []
        events = []
        routing_called = bool(zero.any() or self._cleanup_active)
        if routing_called:
            # A final all-positive observation acknowledges the last deliveries
            # once; after that no dormant routing loop is needed.
            self._sync_positive_placeholders(bins)
            side = self.domain_side / self.grid
            centers = (np.column_stack((bins % self.grid, bins // self.grid)) + .5) * side
            cleanup_goals = self._cleanup.allocate(
                centers, counts=counts, allocation_index=allocation_index)
            details = self._cleanup.last_report
            reserved = np.asarray(details['reserved_counts'], dtype=np.int64)
            for item in details['reservations']:
                identity = item['id']
                if item['kind'] != 'cleanup' or destinations[identity] != -1:
                    raise RuntimeError('Unexpected or duplicated cleanup reservation')
                destinations[identity] = item['destination_bin']
            blocked = copy.deepcopy(details['blocked'])
            routes = copy.deepcopy(details['routes'])
            events = copy.deepcopy(details['events_this_call'])
            self._cleanup_active = bool(zero.any())
        capacity = np.maximum(0, self.quota - counts - reserved)
        # Positive bins with no prescribed population never receive a positive
        # proposal; this is explicit even though their quota is already zero.
        capacity[self.target == 0] = 0
        return dict(counts=counts, zero=zero, cleanup_goals=cleanup_goals,
                    reserved=reserved, destinations=destinations, blocked=blocked,
                    routes=routes, events=events, capacity=capacity,
                    routing_called=routing_called)

    def _report(self, bins, plan, allocation_index, next_bins=None,
                proposed=None, accepted=None, rejection_reasons=None, observation_only=False):
        if next_bins is None:
            next_bins = bins
        proposed = [] if proposed is None else proposed
        accepted = [] if accepted is None else accepted
        rejection_reasons = [] if rejection_reasons is None else rejection_reasons
        after = np.bincount(next_bins, minlength=len(self.target))
        before_hd = float(np.linalg.norm(
            np.sqrt(plan['counts'] / self.population) - np.sqrt(self.target)) / np.sqrt(2))
        after_hd = float(np.linalg.norm(
            np.sqrt(after / self.population) - np.sqrt(self.target)) / np.sqrt(2))
        rejected = [entry['id'] for entry in rejection_reasons]
        cleanup_ids = np.flatnonzero(plan['zero']).tolist()
        report = dict(
            allocation_index=allocation_index, observation_only=observation_only,
            counts_before=plan['counts'].tolist(), counts_after=after.tolist(),
            quota=self.quota.tolist(), reserved_counts=plan['reserved'].tolist(),
            reserved_destinations=plan['destinations'].tolist(),
            positive_admission_capacity=plan['capacity'].tolist(),
            cleanup_goals=plan['cleanup_goals'].tolist(), cleanup_ids=cleanup_ids,
            zero_cleanup_next_hops=[dict(id=i, source_bin=int(bins[i]),
                                        next_bin=int(plan['cleanup_goals'][i])) for i in cleanup_ids],
            positive_proposed_ids=proposed, positive_accepted_ids=accepted,
            positive_rejected_ids=rejected, positive_rejection_reasons=rejection_reasons,
            positive_proposed_count=len(proposed), positive_accepted_count=len(accepted),
            positive_rejected_count=len(rejected), blocked=plan['blocked'],
            cleanup_routes=plan['routes'], cleanup_events=plan['events'],
            routing_called=plan['routing_called'],
            zero_target_population_before=int(plan['zero'].sum()),
            zero_target_population_after=int(after[self.target == 0].sum()),
            hd_before=before_hd, hd_after=after_hd, diagonal_rule=DIAGONAL_RULE,
            positive_target_movement='Original positive-to-positive proposals subject to surplus/deficit and unreserved integer capacity',
            scope='Fixed-population synchronous macroscopic allocation; no physical motion or collision claim')
        self.last_report = report
        return report

    def allocate(self, locations, positive_proposals, allocation_index=None):
        bins = self._actual(locations)
        proposals = self._locations(positive_proposals, 'positive_proposals')
        index = self._index(allocation_index)
        plan = self._cleanup_plan(bins, index)
        next_bins = bins.copy()
        next_bins[plan['zero']] = plan['cleanup_goals'][plan['zero']]
        remaining = plan['capacity'].copy()
        density = np.asarray(plan['counts'] / self.population, dtype=np.float32)
        target = np.asarray(self.target, dtype=np.float32)
        proposed = np.flatnonzero((~plan['zero']) & (proposals != bins)).tolist()
        accepted, rejected = [], []
        for identity in proposed:  # Flatnonzero provides ascending stable IDs.
            source, destination = int(bins[identity]), int(proposals[identity])
            reason = None
            if self.target[destination] == 0:
                reason = 'positive_to_zero_forbidden'
            elif max(abs(source % self.grid - destination % self.grid),
                     abs(source // self.grid - destination // self.grid)) != 1:
                reason = 'not_a_nine_neighbor_connection'
            elif density[source] <= target[source]:
                reason = 'source_not_above_target'
            elif density[destination] >= target[destination]:
                reason = 'destination_not_below_target'
            elif remaining[destination] <= 0:
                reason = 'no_unreserved_integer_capacity'
            if reason is None:
                next_bins[identity] = destination
                remaining[destination] -= 1
                accepted.append(identity)
            else:
                rejected.append(dict(id=identity, source_bin=source,
                                     destination_bin=destination, reason=reason))
        after = np.bincount(next_bins, minlength=len(self.target))
        # Existing excess is allowed to persist/depart; no newly added excess
        # may be created by either the cleanup or positive admission path.
        bound = np.maximum(plan['counts'], self.quota)
        if (after[self.target > 0] > bound[self.target > 0]).any():
            raise RuntimeError('A synchronous allocation created new quota excess')
        if int(after[self.target == 0].sum()) > int(plan['zero'].sum()):
            raise RuntimeError('A synchronous allocation increased zero-target population')
        self._calls += 1
        self._expected_next = next_bins.copy()
        report = self._report(bins, plan, index, next_bins, proposed, accepted, rejected)
        return next_bins, report

    def observe(self, locations, allocation_index=None):
        """Observe endpoint routes/blocked state without executing another hop."""
        bins = self._actual(locations)
        index = self._index(allocation_index)
        plan = self._cleanup_plan(bins, index)
        self._observations += 1
        return self._report(bins, plan, index, observation_only=True)

    def state_dict(self):
        """JSON-safe audit snapshot, with the old router's exact event history."""
        return copy.deepcopy(dict(
            version=1, variant='zero_cleanup_with_positive_target_rebalancing',
            target=self.target.tolist(), population=self.population, grid=self.grid,
            domain_side=self.domain_side, quota=self.quota.tolist(),
            calls=self._calls, observations=self._observations,
            expected_next_locations=None if self._expected_next is None else self._expected_next.tolist(),
            cleanup_routing_active=self._cleanup_active,
            routing_engine_state=self._cleanup.state_dict(),
            old_events=self._cleanup.events,
            current_counts=None if not self.last_report else self.last_report['counts_after'],
            last_report=self.last_report))

    def load_state_dict(self, state):
        """Restore the precise macro continuation, never an identity reordering."""
        try:
            value = json.loads(json.dumps(state, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError('State must be finite JSON data') from error
        for key, expected in [('version', 1), ('variant', 'zero_cleanup_with_positive_target_rebalancing'),
                              ('population', self.population), ('grid', self.grid),
                              ('domain_side', self.domain_side), ('target', self.target.tolist()),
                              ('quota', self.quota.tolist())]:
            if value.get(key) != expected:
                raise ValueError(f'State configuration mismatch: {key}')
        for key in ('calls', 'observations'):
            if not isinstance(value.get(key), int) or value[key] < 0:
                raise ValueError(f'Invalid {key}')
        if not isinstance(value.get('cleanup_routing_active'), bool):
            raise ValueError('Invalid routing-active state')
        expected = value.get('expected_next_locations')
        if expected is not None:
            expected = self._locations(expected, 'expected_next_locations')
        elif value['calls']:
            raise ValueError('Executed calls require expected macro locations')
        self._cleanup.load_state_dict(value['routing_engine_state'])
        if value.get('old_events') != self._cleanup.events:
            raise ValueError('Routing event histories disagree')
        report = value.get('last_report')
        if not isinstance(report, dict):
            raise ValueError('Invalid last_report')
        current = value.get('current_counts')
        if current is not None:
            if expected is None:
                # Observation-only snapshots have no pending macro update.
                if value['calls']:
                    raise ValueError('Missing expected macro locations')
            elif not np.array_equal(current, np.bincount(expected, minlength=len(self.target))):
                raise ValueError('Current counts disagree with expected macro locations')
        self._expected_next = expected
        self._cleanup_active = value['cleanup_routing_active']
        self._calls = value['calls']
        self._observations = value['observations']
        self.last_report = report
        return self

