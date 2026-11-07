"""Independent optional dual-warm-start projection; never compiles on import.

The caller owns row identity mapping. A previous row's scalar dual may be reused
with its CURRENT row normal, provided its sign remains valid for the new slab.
Initialization is x=current_nominal-current_A.T@mapped_dual. In particular, neither
an old primal solution nor a nonzero dual paired with x=nominal is a valid start.
The physical trajectory and braking backup must still be verified by the caller.
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
    # Avoid resolving the WSL-mounted source path on every loaded-kernel call.
    path = Path(library_path) if library_path else Path(__file__).with_name(
        'warm_command_projection_kernel.dll' if sys.platform == 'win32'
        else 'warm_command_projection_kernel.so')
    key = str(path.absolute())
    if key not in _LIBRARIES:
        if not path.is_file():
            raise FileNotFoundError(f'Optional warm projection library is not compiled: {path}')
        library = ctypes.CDLL(key)
        function = library.warm_project_command_slabs
        function.argtypes = ([ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p]*9
                             + [ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_int64, ctypes.c_void_p])
        function.restype = ctypes.c_int
        _LIBRARIES[key] = (library, function)
    return _LIBRARIES[key][1]


def project_command_polytope_warm(nominal, A_csr, lower, upper, dual=None,
                                  max_sweeps=400, primal_tolerance=1e-9,
                                  change_tolerance=1e-9, gap_tolerance=1e-8,
                                  check_every=5, library_path=None):
    """Return (candidate, row_dual, info) for min .5||x-nominal||², l<=Ax<=u.

    ``dual`` is an optional length-row vector mapped to the CURRENT row identities.
    New rows start with zero; deleted rows are dropped. Row order may change only
    when the caller maps it consistently. Upper-only rows require dual>=0,
    lower-only rows require dual<=0, unbounded rows require zero, and finite slabs
    allow either sign. Changing a row normal is allowed: x0 uses the current A.

    The Euclidean norm objective is always centered on the current nominal.
    ``converged`` checks primal, change, stationarity, and absolute complementarity;
    a finite feasible iterate alone is not labeled optimal. Inputs are not mutated.
    """
    start = time.perf_counter()
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim not in (1, 2) or not nominal.size or not np.isfinite(nominal).all():
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
    if np.isposinf(lo).any() or np.isneginf(hi).any() or (lo > hi).any():
        raise ValueError('Inconsistent or infinite-only slab bounds')
    mapped = np.zeros(rows, dtype=np.float64) if dual is None else np.asarray(dual, dtype=np.float64)
    if mapped.shape != (rows,) or not np.isfinite(mapped).all():
        raise ValueError('dual must be a finite length-row vector mapped to current rows')
    invalid = ((np.isposinf(hi) & (mapped > 0)) |
               (np.isneginf(lo) & (mapped < 0)))
    if invalid.any():
        raise ValueError('Mapped dual sign is invalid for the current one-sided/unbounded slab')
    arrays = [np.ascontiguousarray(matrix.indptr, dtype=np.int64),
              np.ascontiguousarray(matrix.indices, dtype=np.int64),
              np.ascontiguousarray(matrix.data, dtype=np.float64),
              np.ascontiguousarray(lo), np.ascontiguousarray(hi),
              np.ascontiguousarray(nominal.ravel()), np.ascontiguousarray(mapped),
              np.empty(nominal.size, dtype=np.float64), np.empty(rows, dtype=np.float64)]
    stats = np.zeros(10, dtype=np.float64)
    code = _kernel(library_path)(nominal.size, rows,
              *[ctypes.c_void_p(a.ctypes.data) for a in arrays], max_sweeps,
              primal_tolerance, change_tolerance, gap_tolerance, check_every,
              ctypes.c_void_p(stats.ctypes.data))
    if code == 1:
        raise ValueError('Constraint intersection contains an inconsistent zero row')
    if code == 3:
        raise ValueError('Native validation rejected a mapped dual outside its domain')
    candidate, row_dual = arrays[-2].reshape(nominal.shape), arrays[-1]
    diagnostics = dict(method='CSR slab dual-warm-start Dykstra projection',
        native_return_code=int(code), converged=bool(code == 0 and stats[5]),
        sweeps=int(stats[0]), primal_residual=float(stats[1]),
        sweep_change=float(stats[2]), duality_gap=float(stats[3]),
        objective=float(stats[4]), native_seconds=float(stats[6]),
        stationarity_residual=float(stats[7]),
        absolute_complementarity=float(stats[8]),
        normalized_complementarity=float(stats[9]), seconds=time.perf_counter()-start)
    if code != 0 or not np.isfinite(candidate).all() or not np.isfinite(row_dual).all():
        diagnostics.update(reason='native_numeric_failure', converged=False)
    else:
        diagnostics['reason'] = 'converged' if diagnostics['converged'] else 'iteration_limit'
    return candidate, row_dual, diagnostics

