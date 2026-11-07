"""Optional compiled polygon arithmetic, preserving the SAME SciPy Voronoi.

No new grouping, neighbor approximation, cached cells or modified target/stop
rules. The native code receives SciPy's unchanged vertices and region order.
Near caller-supplied decision distances, the original Python cell calculation
is used to avoid changing a threshold branch due to centroid roundoff.
Compilation is an explicit external step, never performed by this module.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import Voronoi
from motion_utils import clip_polygon

_LIBRARIES = {}
_DEFAULT_LIBRARY = Path(__file__).with_name(
    'native_voronoi_centroids_kernel.dll' if sys.platform == 'win32'
    else 'native_voronoi_centroids_kernel.so').resolve()


def _kernel(library_path=None):
    # Resolve the default only once. Repeating realpath for every small bin
    # group is disproportionately expensive on a WSL-mounted Windows directory.
    path = Path(library_path).resolve() if library_path else _DEFAULT_LIBRARY
    key = str(path)
    if key not in _LIBRARIES:
        if not path.is_file():
            raise FileNotFoundError(f'Optional centroid library has not been compiled: {path}')
        library = ctypes.CDLL(key)
        function = library.clipped_voronoi_centroids
        function.argtypes = ([ctypes.c_int64]*2+[ctypes.c_void_p]*3
                             +[ctypes.c_double]*4+[ctypes.c_void_p]*3)
        function.restype = ctypes.c_int
        _LIBRARIES[key] = (library, function)
    return _LIBRARIES[key][1]


def _original_cell(vertices, region, xmin, xmax, ymin, ymax):
    """The original clip and NumPy shoelace arithmetic, without another Qhull."""
    polygon = vertices[region]
    for normal, bound in [(np.array([1., 0]), xmax), (np.array([-1., 0]), -xmin),
                          (np.array([0., 1]), ymax), (np.array([0., -1]), -ymin)]:
        polygon = clip_polygon(polygon, normal, bound)
    nxt = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0]*nxt[:, 1]-polygon[:, 1]*nxt[:, 0]
    if len(polygon) < 3 or abs(cross.sum()) < 1e-12:
        raise RuntimeError('Degenerate clipped Voronoi cell')
    return ((polygon+nxt)*cross[:, None]).sum(0)/(3*cross.sum())


def native_voronoi_centroids(points, x_min, x_max, y_min, y_max,
                             decision_distances=(.1,), grey_zone_m=1e-9,
                             library_path=None, return_diagnostics=False):
    """Drop-in centroid array, with optional numerical-decision diagnostics.

    The default decision distance is the unchanged nominal 0.1 m stop threshold.
    Callers applying additional distance thresholds can pass those too, e.g.
    ``decision_distances=(.1, 2*tau*vmax)`` for the arrival-aware command layer.
    A grey-zone fallback re-evaluates the ORIGINAL centroid, not a new rule.
    """
    started = time.perf_counter()
    points = np.asarray(points, dtype=np.float64)
    bounds = np.asarray([x_min, x_max, y_min, y_max], dtype=np.float64)
    decisions = np.asarray(decision_distances, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty N-by-2 points required')
    if not np.isfinite(bounds).all() or x_max <= x_min or y_max <= y_min:
        raise ValueError('Finite ordered cell bounds required')
    if decisions.ndim != 1 or not np.isfinite(decisions).all() or (decisions < 0).any():
        raise ValueError('Finite nonnegative decision distances required')
    if not np.isfinite(grey_zone_m) or grey_zone_m < 0:
        raise ValueError('Finite nonnegative grey-zone width required')
    if len(points) == 1:
        result = np.array([[(x_min+x_max)/2, (y_min+y_max)/2]])
        diagnostics = dict(points=1, python_fallback_cells=0, decision_grey_cells=0,
                           native_seconds=0., qhull_seconds=0., maximum_cancellation_ratio=0.,
                           seconds=time.perf_counter()-started)
        return (result, diagnostics) if return_diagnostics else result
    boundary = [[x_min-10000, y_min-10000], [x_min-10000, y_max+10000],
                [x_max+10000, y_min-10000], [x_max+10000, y_max+10000]]
    before_qhull = time.perf_counter()
    vor = Voronoi(np.vstack([points, boundary]))
    qhull_seconds = time.perf_counter()-before_qhull
    regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]
    if any(-1 in region for region in regions):
        raise RuntimeError('Auxiliary boundary points failed to bound cell')
    lengths = np.asarray([len(region) for region in regions], dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]
    indices = np.ascontiguousarray(np.concatenate([np.asarray(region, dtype=np.int64) for region in regions]))
    vertices = np.ascontiguousarray(vor.vertices, dtype=np.float64)
    result = np.empty((len(points), 2), dtype=np.float64)
    fallback = np.zeros(len(points), dtype=np.int32)
    stats = np.zeros(2, dtype=np.float64)
    inputs = [vertices, offsets, indices]
    outputs = [result, fallback, stats]
    code = _kernel(library_path)(len(points), len(vertices),
                                *[ctypes.c_void_p(value.ctypes.data) for value in inputs],
                                *bounds, *[ctypes.c_void_p(value.ctypes.data) for value in outputs])
    if code:
        raise RuntimeError(f'Invalid native Voronoi region index: {code}')
    distance = np.linalg.norm(result-points, axis=1)
    grey = (np.any(np.abs(distance[:, None]-decisions[None, :]) <= grey_zone_m, axis=1)
            if len(decisions) else np.zeros(len(points), dtype=bool))
    recheck = (fallback != 0) | grey
    for i in np.flatnonzero(recheck):
        result[i] = _original_cell(vor.vertices, regions[i], *bounds)
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite clipped Voronoi centroid')
    diagnostics = dict(points=len(points), python_fallback_cells=int(recheck.sum()),
                       decision_grey_cells=int(grey.sum()), native_seconds=float(stats[1]),
                       qhull_seconds=qhull_seconds, maximum_cancellation_ratio=float(stats[0]),
                       seconds=time.perf_counter()-started)
    return (result, diagnostics) if return_diagnostics else result

