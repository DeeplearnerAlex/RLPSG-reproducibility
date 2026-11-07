"""Synthetic bin-level RLPSG demo; not a paper-data replay or motion simulation.

This example creates its own initial robot bins. It demonstrates frozen-PPO
probabilistic allocation and current-HD terminal switching. Bin hops execute
instantaneously here; SFVOAPF motion execution is a separate integration layer.
There is no relative-position adjustment in this bin-only example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from terminal_positive_rebalance import TerminalPositiveRebalanceAllocator
from policy import MeanActionPolicy, hellinger_distance, positive_support_matrix


def sample_bins(matrix, locations, rng):
    """One local RNG draw per identity, ascending destination-bin CDF."""
    probabilities = np.asarray(matrix, dtype=np.float64)
    probabilities = probabilities/probabilities.sum(axis=0, keepdims=True)
    cumulative = np.cumsum(probabilities[:, locations], axis=0)
    cumulative[-1] = 1.
    return np.sum(rng.random(len(locations))[None, :] > cumulative, axis=0).astype(np.int64)


def run(policy, robots=2000, steps=30, seed=0, initial_bin=0):
    if (isinstance(robots, bool) or robots < 1 or isinstance(steps, bool) or steps < 0
            or seed < 0 or not 0 <= initial_bin < 64):
        raise ValueError('Use robots>=1, steps>=0, seed>=0, and initial-bin in 0..63')
    rng = np.random.default_rng(seed)
    locations = np.full(robots, initial_bin, dtype=np.int64)
    counts = np.empty((steps+1, 64), dtype=np.int64)
    hd = np.empty(steps+1, dtype=np.float64)
    mode = np.empty(steps+1, dtype=bool)
    moves = np.zeros(steps, dtype=np.int64)
    allocator = None
    entries, exits = [], []
    previous_mode = False
    for endpoint in range(steps+1):
        counts[endpoint] = np.bincount(locations, minlength=64)
        density = counts[endpoint]/robots
        hd[endpoint] = hellinger_distance(density, policy.target)
        mode[endpoint] = hd[endpoint] < .05
        if mode[endpoint] and not previous_mode:
            entries.append(endpoint)
            allocator = TerminalPositiveRebalanceAllocator(policy.target, robots)
        elif not mode[endpoint] and previous_mode:
            exits.append(endpoint)
            allocator = None
        previous_mode = bool(mode[endpoint])
        if endpoint == steps:
            break
        matrix = policy.transition_matrix(density)
        if mode[endpoint]:
            matrix = positive_support_matrix(matrix, policy.target)
        proposals = sample_bins(matrix, locations, rng)
        if allocator is None:
            following = proposals
        else:
            following, _ = allocator.allocate(locations, proposals, allocation_index=endpoint)
        moves[endpoint] = np.count_nonzero(following != locations)
        locations = following
    zero = counts[:, policy.target == 0].sum(axis=1)
    summary = dict(scope='New synthetic bin-level demonstration; no physical motion or paper-data replay',
        robots=int(robots), allocation_updates=int(steps), seed=int(seed), initial_bin=int(initial_bin),
        mode_rule='current_HD_strictly_below_0.05', initial_hd=float(hd[0]), final_hd=float(hd[-1]),
        final_zero_target_population=int(zero[-1]), cumulative_bin_changes=int(moves.sum()),
        final_hd_below_005=bool(hd[-1] < .05),
        final_hd_below_005_and_zero_empty=bool(hd[-1] < .05 and zero[-1] == 0),
        terminal_entries=entries, terminal_exits=exits,
        final_mode='terminal' if mode[-1] else 'probabilistic',
        relative_position_adjustment=False, physical_execution=False)
    return summary, dict(counts=counts, hd=hd, mode=mode, moves=moves,
                         zero_target_population=zero, target=policy.target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('ring', 'letter_A', 'separated'), default='ring')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--robots', type=int, default=2000)
    parser.add_argument('--initial-bin', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Optional NEW directory for this newly generated demo only')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('--output must name a new directory; existing content is never overwritten')
    policy = MeanActionPolicy(args.models_dir / (args.target + '.npz'))
    summary, arrays = run(policy, args.robots, args.steps, args.seed, args.initial_bin)
    summary['target'] = args.target
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / 'demo.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2)
            stream.write('\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
