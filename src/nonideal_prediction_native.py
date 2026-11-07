"""Opt-in native implementation of the existing state-only prediction.

The declared discrete model, clipping, substep count and arithmetic order are
unchanged. This never replaces the truth world's arc-producing integrator.
The library must be compiled explicitly without fast-math or FMA contraction;
importing this module does not compile code, select a command, or mutate a run.

NativeStateOnlyVelocityResponse inherits the existing parameter validation and
keeps the StateOnlyVelocityResponse.advance input checks verbatim before native
arithmetic. The upstream method exposes no separate validation hook.
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

from nonideal_prediction_fast import StateOnlyVelocityResponse
from reduced_velocity_dynamics import VelocityResponse


_DEFAULT_LIBRARY = Path(__file__).with_name(
    'nonideal_prediction_native_kernel_20261105.dll' if sys.platform == 'win32'
    else 'nonideal_prediction_native_kernel_20261105.so').resolve()
_LIBRARIES = {}


def _kernel(library_path=None):
    path = Path(library_path).resolve() if library_path is not None else _DEFAULT_LIBRARY
    key = str(path)
    if key not in _LIBRARIES:
        if not path.is_file():
            raise FileNotFoundError(f'Optional native predictor has not been compiled: {path}')
        library = ctypes.CDLL(key)
        function = library.rlpsg_state_only_prediction_v1
        function.argtypes = [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_double] * 4 + [ctypes.c_void_p] * 3
        function.restype = ctypes.c_int
        _LIBRARIES[key] = (library, function)
    return _LIBRARIES[key][1]


class NativeStateOnlyVelocityResponse(StateOnlyVelocityResponse):
    """Same public predictor/configuration, native per-robot arithmetic only."""

    def advance(self, positions, velocities, command, duration_s):
        # Keep these checks/order identical to StateOnlyVelocityResponse. Use C
        # layout explicitly, since callers may supply strided or Fortran arrays.
        p, v, command = map(lambda x: np.array(x, dtype=float, copy=True), (positions, velocities, command))
        if p.shape != v.shape or command.shape != p.shape or p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Expected equal N-by-2 arrays')
        if not len(p) or not np.isfinite(duration_s) or not np.isfinite([p, v, command]).all() or duration_s <= 0:
            raise ValueError('Finite states and positive duration required')
        if np.linalg.norm(v, axis=1).max() > self.speed_cap_m_s+1e-9:
            raise ValueError('Initial velocity exceeds speed cap')
        p, v, command = map(np.ascontiguousarray, (p, v, command))
        steps = int(np.ceil(duration_s/self.integration_dt_s))
        h = duration_s/steps
        code = _kernel()(len(p), steps, h, self.tau_s, self.speed_cap_m_s,
                         self.acceleration_cap_m_s2,
                         *[ctypes.c_void_p(array.ctypes.data) for array in (p, v, command)])
        if code:
            raise RuntimeError(f'Invalid native state-only prediction arguments: {code}')
        return p, v, []


def install_native_state_only_prediction(network, belief=None):
    """Explicitly replace predictor references only, retaining dataclass config.

    Resolve the library and validate both old references before replacing
    either. A missing build or invalid belief cannot partially install a model.
    """
    if not isinstance(network._model, VelocityResponse):
        raise TypeError('Expected the explicit network matched-model predictor')
    if belief is not None and not isinstance(belief.model, VelocityResponse):
        raise TypeError('Expected the explicit bounded-belief matched model')
    _kernel()
    network_model = NativeStateOnlyVelocityResponse(**asdict(network._model))
    belief_model = None if belief is None else NativeStateOnlyVelocityResponse(**asdict(belief.model))
    network._model = network_model
    if belief is not None:
        belief.model = belief_model

