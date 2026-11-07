"""Relative-position-aware sampling and coefficient fitting for an 8x8 grid.

These pure method functions retain the project arithmetic without importing an
experiment driver or loading any experimental records.
"""
import numpy as np
from scipy.optimize import least_squares

from motion_utils import bins_and_counts


def hd_from_counts(counts, phi, count):
    # Out-of-domain agents are an extra category with target probability zero.
    outside = max(0, count-int(counts.sum())) / count
    return float(np.sqrt((np.sum((np.sqrt(counts/count)-np.sqrt(phi))**2)+outside)/2))


def sample_goals(points, matrix, seed, allocation_step, quadrature_side=48):
    """Nine-neighbor position adjustment with fitted spatial-average weights.

    A neighbor is represented by the nearest boundary midpoint/corner of the
    source bin; the self-transition reference is the bin center. Bin numbering
    uses x=column, y=row consistently with physical positions and policy counts.
    """
    bins, _, outside = bins_and_counts(points)
    if outside:
        raise ValueError('Cannot assign an out-of-domain robot')
    rng = np.random.default_rng(np.random.SeedSequence([seed, allocation_step, 9107]))
    uniforms = rng.random(len(points))  # Same indexed draws for all paired cases.
    goals = np.empty(len(points), dtype=int)
    worst = 0.
    z = (np.arange(quadrature_side)+.5)*100/quadrature_side
    xx, yy = np.meshgrid(z, z)
    quad = np.column_stack((xx.ravel(), yy.ravel()))
    for source in np.unique(bins):
        indices = np.flatnonzero(bins == source)
        column = matrix[:, source].astype(float)
        active = np.flatnonzero(column > 0)
        target = column[active]/column[active].sum()
        if len(active) == 1:
            goals[indices] = active[0]
            continue
        offset = np.column_stack((active % 8-source % 8, active//8-source//8))
        if abs(offset).max() > 1:
            raise ValueError('Policy matrix has a nonlocal edge')
        reference = 50+50*offset
        distance = np.maximum(np.linalg.norm(quad[:, None]-reference[None], axis=2), 1e-6)

        def probability(d, values):
            log_weight = np.log(target) + np.r_[values, 0.]
            log_weight -= log_weight.max()
            w = np.exp(log_weight)/d
            return w/w.sum(axis=1, keepdims=True)

        def residual(values):
            return (probability(distance, values).mean(axis=0)-target)[:-1]

        fit = least_squares(residual, np.zeros(len(active)-1), bounds=(-20, 20),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
        error = float(abs(probability(distance, fit.x).mean(axis=0)-target).max())
        if not fit.success or error > 1e-5:
            raise RuntimeError(f'Balancing failure: bin={source}, residual={error}')
        worst = max(worst, error)
        local = points[indices]-np.array([source % 8, source//8])*100
        d = np.maximum(np.linalg.norm(local[:, None]-reference[None], axis=2), 1e-6)
        prob = probability(d, fit.x)
        selected = (uniforms[indices, None] > prob.cumsum(axis=1)).sum(axis=1)
        goals[indices] = active[np.minimum(selected, len(active)-1)]
    return goals, worst

