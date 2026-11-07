"""Portable NumPy CPU mean-action inference for the released RLPSG actors.

Only actor parameters and frozen observation statistics are loaded. This module
does not import a trainer, restore an optimizer, read experimental records, or
update normalization. Matrix entries follow destination-row/source-column order.
NumPy/BLAS and Flax/JAX need not be bitwise identical; the model exporter should
check their numerical agreement before publishing the exported actor files.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def hellinger_distance(density, target):
    """Unsmoothed, float64 HD used by the allocation-mode decision."""
    rho, phi = np.asarray(density, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (rho.shape != (64,) or phi.shape != rho.shape or not np.isfinite([rho, phi]).all()
            or (rho < 0).any() or (phi < 0).any() or rho.sum() <= 0 or phi.sum() <= 0):
        raise ValueError('Expected finite nonnegative 64-bin distributions')
    return float(np.linalg.norm(np.sqrt(rho/rho.sum()) - np.sqrt(phi/phi.sum()))/np.sqrt(2.))


def positive_support_matrix(matrix, target):
    """Condition positive-source columns on positive destinations before sampling.

    Zero-source columns are retained; the terminal router replaces those goals.
    No new inter-bin edge is created. Empty columns receive unit self probability.
    """
    result = np.array(matrix, dtype=np.float64, copy=True)
    phi = np.asarray(target, dtype=np.float64)
    if (result.shape != (64, 64) or phi.shape != (64,) or
            not np.isfinite(result).all() or (result < 0).any()):
        raise ValueError('Expected a nonnegative 64-by-64 transition matrix')
    positive = phi > 0
    result[np.ix_(~positive, positive)] = 0.
    empty = np.flatnonzero(result.sum(axis=0) == 0)
    result[empty, empty] = 1.
    result /= result.sum(axis=0, keepdims=True)
    return result


class MeanActionPolicy:
    """Frozen 65→512→512→512→274 actor and its allocation transformation.

    NPZ format version 1 contains only ``target``, ``edge_pairs``,
    ``obs_mean``, ``obs_variance``, ``dense_{0..3}_kernel/bias`` and
    ``ln_{0..2}_scale/bias``, plus scalar ``format_version=1``. Dense kernels
    use Flax's (input, output) layout. LayerNorm uses epsilon 1e-6 and the
    fast variance E[x²]-E[x]²; GELU uses the tanh approximation. Source and
    destination gates retain the original float32 density/target precision.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        with np.load(self.model_path, allow_pickle=False) as source:
            if int(np.asarray(source['format_version']).item()) != 1:
                raise ValueError('Unsupported actor NPZ format_version')
            self.target32 = np.array(source['target'], dtype=np.float32, copy=True)
            self.edge_pairs = np.array(source['edge_pairs'], dtype=np.int64, copy=True)
            self.obs_mean = np.array(source['obs_mean'], dtype=np.float32, copy=True)
            self.obs_variance = np.array(source['obs_variance'], dtype=np.float32, copy=True)
            self.kernels = [np.array(source[f'dense_{i}_kernel'], dtype=np.float32, copy=True) for i in range(4)]
            self.biases = [np.array(source[f'dense_{i}_bias'], dtype=np.float32, copy=True) for i in range(4)]
            self.scales = [np.array(source[f'ln_{i}_scale'], dtype=np.float32, copy=True) for i in range(3)]
            self.offsets = [np.array(source[f'ln_{i}_bias'], dtype=np.float32, copy=True) for i in range(3)]
        if (self.target32.shape != (64,) or not np.isfinite(self.target32).all()
                or (self.target32 < 0).any() or not np.isclose(self.target32.sum(dtype=np.float64), 1., atol=1e-6)):
            raise ValueError('Invalid target distribution in actor file')
        if (self.obs_mean.shape != (65,) or self.obs_variance.shape != (65,)
                or not np.isfinite([self.obs_mean, self.obs_variance]).all()
                or (self.obs_variance < 0).any()):
            raise ValueError('Invalid frozen observation statistics')
        expected_pairs = np.array([(i, j) for i in range(64) for j in range(i+1, 64)
            if max(abs(i % 8-j % 8), abs(i//8-j//8)) == 1], dtype=np.int64)
        if not np.array_equal(self.edge_pairs, expected_pairs):
            raise ValueError('edge_pairs must be the 210 lexicographic undirected eight-neighbor pairs')
        dimensions = (65, 512, 512, 512, 274)
        for index in range(4):
            if (self.kernels[index].shape != (dimensions[index], dimensions[index+1])
                    or self.biases[index].shape != (dimensions[index+1],)
                    or not np.isfinite(self.kernels[index]).all() or not np.isfinite(self.biases[index]).all()):
                raise ValueError(f'Invalid dense layer {index}')
        for index in range(3):
            if (self.scales[index].shape != (512,) or self.offsets[index].shape != (512,)
                    or not np.isfinite([self.scales[index], self.offsets[index]]).all()):
                raise ValueError(f'Invalid LayerNorm layer {index}')
        self.target = self.target32.astype(np.float64)
        self.target /= self.target.sum()
        for array in ([self.target, self.target32, self.edge_pairs, self.obs_mean, self.obs_variance]
                      + self.kernels + self.biases + self.scales + self.offsets):
            array.flags.writeable = False

    @staticmethod
    def _density(value):
        rho = np.asarray(value, dtype=np.float32)
        if (rho.shape != (64,) or not np.isfinite(rho).all() or (rho < 0).any()
                or not np.isclose(rho.sum(dtype=np.float64), 1., rtol=0., atol=1e-5)):
            raise ValueError('density must be a finite unit-sum 64-vector')
        return rho

    def mean_action(self, density):
        rho = self._density(density)
        eps = np.float32(1e-8)
        difference = np.sqrt(rho+eps)-np.sqrt(self.target32+eps)
        hd = np.sqrt(np.sum(difference*difference, dtype=np.float32)/np.float32(2.))
        observation = np.concatenate((rho, np.asarray([hd], dtype=np.float32)))
        value = (observation-self.obs_mean)/np.sqrt(self.obs_variance+eps)
        for index in range(3):
            value = value @ self.kernels[index] + self.biases[index]
            mean = np.mean(value, dtype=np.float32)
            variance = np.maximum(np.float32(0.), np.mean(value*value, dtype=np.float32)-mean*mean)
            value = ((value-mean)/np.sqrt(variance+np.float32(1e-6)))*self.scales[index]+self.offsets[index]
            value = np.float32(.5)*value*(np.float32(1.)+np.tanh(np.float32(np.sqrt(2./np.pi))*(
                value+np.float32(.044715)*value*value*value)))
        action = value @ self.kernels[3] + self.biases[3]
        if action.shape != (274,) or not np.isfinite(action).all():
            raise FloatingPointError('Actor produced nonfinite or incorrectly shaped actions')
        return np.asarray(action, dtype=np.float32)

    def transition_matrix(self, density):
        """Mean action → source gate → weighting → receiving gate → normalization."""
        rho = self._density(density)
        action = self.mean_action(rho)
        weights = np.zeros((64, 64), dtype=np.float32)
        first, second = self.edge_pairs.T
        weights[first, second] = np.maximum(action[:210], np.float32(0.))
        weights[second, first] = np.maximum(-action[:210], np.float32(0.))
        diagonal = np.arange(64)
        weights[diagonal, diagonal] = np.abs(action[210:])
        source_allowed = rho > self.target32
        weights *= source_allowed[None, :]
        weights[diagonal[~source_allowed], diagonal[~source_allowed]] += np.float32(1.)
        # The original destination masks are boolean; their addition is a
        # logical union, not an extra numeric weight for zero-target bins.
        empty = np.flatnonzero(weights.sum(axis=0, dtype=np.float32) == 0)
        weights[empty, empty] = np.float32(1.)
        weights /= weights.sum(axis=0, keepdims=True, dtype=np.float32)
        receiving = (rho < self.target32) | (self.target32 == 0)
        allowed = receiving[:, None] | np.eye(64, dtype=bool)
        weights = np.where(allowed, weights, np.float32(0.))
        empty = np.flatnonzero(weights.sum(axis=0, dtype=np.float32) == 0)
        weights[empty, empty] = np.float32(1.)
        weights /= weights.sum(axis=0, keepdims=True, dtype=np.float32)
        if (not np.isfinite(weights).all() or (weights < 0).any()
                or np.max(np.abs(weights.sum(axis=0)-1.)) > 1e-5):
            raise FloatingPointError('Generated matrix is not column stochastic')
        return weights
