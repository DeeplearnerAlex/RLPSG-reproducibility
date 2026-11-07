"""Optional lazy native CSR-slab Dykstra solver; never compiles on import.

The solution is the projection of the CURRENT nominal command. Every call starts
with x=nominal and zero row corrections; no stale objective or dual warm start.
The caller must still verify its actual physical trajectory and braking backup.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

_LIBRARIES = {}


def _kernel(library_path=None):
    path = Path(library_path) if library_path else Path(__file__).with_name(
        'command_projection_kernel.dll' if sys.platform == 'win32' else 'command_projection_kernel.so')
    path = path.resolve()
    key = str(path)
    if key not in _LIBRARIES:
        if not path.is_file():
            raise FileNotFoundError(f'Optional projection library is not compiled: {path}')
        library = ctypes.CDLL(key)
        function = library.project_command_slabs
        function.argtypes = ([ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p]*7
                             + [ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_int64, ctypes.c_void_p])
        function.restype = ctypes.c_int
        _LIBRARIES[key] = (library, function)
    return _LIBRARIES[key][1]


def project_command_polytope(nominal, A_csr, lower, upper, max_sweeps=400,
                             primal_tolerance=1e-9, change_tolerance=1e-9,
                             gap_tolerance=1e-8, check_every=5, library_path=None):
    """Return (candidate, diagnostics) for min .5||x-nominal||², l<=A x<=u.

    Nominal may be a vector or N-by-2 array; the result preserves its shape.
    ``converged`` checks primal feasibility, inter-sweep change, stationarity and
    normalized absolute complementarity. Merely feasible finite iterates are not
    mislabeled optimal. No mutation of inputs and no automatic shell/compiler.
    """
    start = time.perf_counter()
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim not in [1, 2] or not nominal.size or not np.isfinite(nominal).all():
        raise ValueError('nominal must be a nonempty finite vector or matrix')
    if not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
        raise ValueError('max_sweeps must be a positive integer')
    if not isinstance(check_every, (int, np.integer)) or check_every < 1:
        raise ValueError('check_every must be a positive integer')
    tolerances = [primal_tolerance, change_tolerance, gap_tolerance]
    if not np.isfinite(tolerances).all() or min(tolerances) < 0:
        raise ValueError('finite nonnegative tolerances required')
    if not sparse.issparse(A_csr) or np.iscomplexobj(A_csr.data):
        raise ValueError('A_csr must be a real scipy sparse matrix')
    matrix = A_csr.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.shape[1] != nominal.size or not np.isfinite(matrix.data).all():
        raise ValueError('Matrix dimensions/data do not match nominal')
    rows = matrix.shape[0]
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.shape != (rows,) or hi.shape != (rows,) or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError('lower/upper must be length-row arrays without NaNs')
    if np.isposinf(lo).any() or np.isneginf(hi).any() or (lo>hi).any():
        raise ValueError('Inconsistent or infinite-only slab bounds')
    arrays = [np.ascontiguousarray(matrix.indptr, dtype=np.int64),
              np.ascontiguousarray(matrix.indices, dtype=np.int64),
              np.ascontiguousarray(matrix.data, dtype=np.float64),
              np.ascontiguousarray(lo), np.ascontiguousarray(hi),
              np.ascontiguousarray(nominal.ravel()), np.empty(nominal.size, dtype=np.float64)]
    stats = np.zeros(10, dtype=np.float64)
    code = _kernel(library_path)(nominal.size, rows,
              *[ctypes.c_void_p(a.ctypes.data) for a in arrays], max_sweeps,
              primal_tolerance, change_tolerance, gap_tolerance, check_every,
              ctypes.c_void_p(stats.ctypes.data))
    if code == 1:
        raise ValueError('Constraint intersection contains an inconsistent zero row')
    candidate = arrays[-1].reshape(nominal.shape)
    diagnostics = dict(method='CSR slab Dykstra projection', native_return_code=int(code),
        converged=bool(code == 0 and stats[5]), sweeps=int(stats[0]),
        primal_residual=float(stats[1]), sweep_change=float(stats[2]),
        duality_gap=float(stats[3]), objective=float(stats[4]),
        native_seconds=float(stats[6]), stationarity_residual=float(stats[7]),
        absolute_complementarity=float(stats[8]), normalized_complementarity=float(stats[9]),
        seconds=time.perf_counter()-start)
    if code != 0 or not np.isfinite(candidate).all():
        diagnostics['reason'] = 'native_numeric_failure'
        diagnostics['converged'] = False
    else:
        diagnostics['reason'] = 'converged' if diagnostics['converged'] else 'iteration_limit'
    return candidate, diagnostics

