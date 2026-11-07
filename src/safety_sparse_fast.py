"""Optional exact sparse bookkeeping helpers; no controller or solver is changed.

These helpers never mutate a supplied matrix or warm state.  Column order, explicit
zeros, duplicate entries and the scalar dual values are retained.  Unusual sparse
indexing falls back to the original SciPy expression rather than being redefined.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _integer_vector(value):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in 'iu':
        return None
    if value.dtype.kind == 'u' and value.size and int(value.max()) > np.iinfo(np.int64).max:
        return None
    return value.astype(np.int64, copy=False)


class RowDualLookup:
    """Snapshot the old ``{int(k): float(d) for k,d in zip(...)}`` lookup.

    Zip truncation and last-duplicate-wins are intentional.  ``get`` returns a new
    float64 array, including on its unchanged-key fast path.  The caller may apply
    its existing one-sided dual-sign mask afterwards without modifying this cache.
    """
    def __init__(self, state, population):
        self._dictionary = None
        self._original_keys = self._original_dual = None
        self._keys = np.empty(0, dtype=np.int64)
        self._dual = np.empty(0, dtype=np.float64)
        if state.get('population') != population:
            return
        incoming_keys = state.get('row_keys', [])
        incoming_dual = state.get('dual', [])
        keys = _integer_vector(incoming_keys)
        dual = np.asarray(incoming_dual)
        if keys is None or dual.ndim != 1 or dual.dtype.kind not in 'biuf':
            self._dictionary = {int(k): float(d) for k, d in zip(incoming_keys, incoming_dual)}
            return
        count = min(len(keys), len(dual))
        self._original_keys = keys[:count].copy()
        self._original_dual = dual[:count].astype(np.float64, copy=True)
        if not count:
            return
        order = np.argsort(self._original_keys, kind='stable')
        sorted_keys = self._original_keys[order]
        last = np.r_[sorted_keys[1:] != sorted_keys[:-1], True]
        self._keys = sorted_keys[last]
        self._dual = self._original_dual[order[last]]
        # Repeated original keys must all receive their LAST value, not raw dual.
        self._unchanged_unique = len(self._keys) == count

    def get(self, keys):
        integer_keys = _integer_vector(keys)
        if self._dictionary is not None:
            return np.array([self._dictionary.get(int(key), 0.) for key in keys], dtype=np.float64)
        if integer_keys is None:
            dictionary = {int(k): float(d) for k, d in zip(self._keys, self._dual)}
            return np.array([dictionary.get(int(key), 0.) for key in keys], dtype=np.float64)
        if (self._original_keys is not None and getattr(self, '_unchanged_unique', False)
                and np.array_equal(integer_keys, self._original_keys)):
            return self._original_dual.copy()
        result = np.zeros(len(integer_keys), dtype=np.float64)
        if not len(self._keys):
            return result
        positions = np.searchsorted(self._keys, integer_keys)
        within = positions < len(self._keys)
        found = np.flatnonzero(within)
        found = found[self._keys[positions[found]] == integer_keys[found]]
        result[found] = self._dual[positions[found]]
        return result


def component_csr(matrix, rows, columns):
    """Equivalent CSR component of ``matrix[rows][:, columns]``.

    The optimized path requires nonnegative one-dimensional integer indices and
    strictly increasing columns (the controller's component-index convention).
    All other index semantics, including reordered/repeated/Boolean columns, use
    SciPy itself.  A complete-range result may alias the input: treat it read-only.
    """
    row_ids, column_ids = _integer_vector(rows), _integer_vector(columns)
    if (not sparse.issparse(matrix) or matrix.format != 'csr'
            or row_ids is None or column_ids is None or not len(column_ids)
            or (row_ids < 0).any() or (row_ids >= matrix.shape[0]).any()
            or (column_ids < 0).any() or (column_ids >= matrix.shape[1]).any()
            or (np.diff(column_ids) <= 0).any()):
        return matrix[rows][:, columns]
    all_rows = len(row_ids) == matrix.shape[0] and np.array_equal(row_ids, np.arange(matrix.shape[0]))
    all_columns = len(column_ids) == matrix.shape[1] and np.array_equal(column_ids, np.arange(matrix.shape[1]))
    if all_rows and all_columns:
        return matrix
    part = matrix if all_rows else matrix[rows]
    if all_columns:
        return part
    local = np.searchsorted(column_ids, part.indices)
    included = local < len(column_ids)
    ids = np.flatnonzero(included)
    included[ids] = column_ids[local[ids]] == part.indices[ids]
    if included.all():
        data, indptr = part.data, part.indptr
    else:
        data = part.data[included]
        prefix = np.r_[0, np.cumsum(included, dtype=np.int64)]
        indptr = prefix[part.indptr].astype(part.indptr.dtype, copy=False)
    indices = local[included].astype(part.indices.dtype, copy=False)
    cls = sparse.csr_array if isinstance(matrix, sparse.csr_array) else sparse.csr_matrix
    result = cls((data, indices, indptr), shape=(len(row_ids), len(column_ids)), copy=False)
    # Preserve the existing index precision even when SciPy's constructor can
    # safely downcast small arrays; ordinary fancy slicing retains that precision.
    result.indices = indices
    result.indptr = indptr
    return result

