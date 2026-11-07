"""Optional compiled batch swept checker; no shell compilation on import.

Uses the same continuous piecewise-parabolic trajectory and overlap tolerance
as accelerated_arc_batch_metrics. The native narrow phase solves cubic extrema
by derivative critical-point partition and bisection rather than np.roots.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

_LIBRARIES={}


def _kernel(library_path=None):
    path=Path(library_path) if library_path else Path(__file__).with_name(
        'accelerated_arc_batch_kernel.dll' if sys.platform=='win32' else 'accelerated_arc_batch_kernel.so')
    path=path.resolve(); key=str(path)
    if key not in _LIBRARIES:
        if not path.is_file():
            raise FileNotFoundError(f'Optional arc library has not been compiled: {path}')
        library=ctypes.CDLL(key); fn=library.accelerated_arc_metrics
        fn.argtypes=([ctypes.c_int64]*3+[ctypes.c_void_p]*5+[ctypes.c_double]*3+[ctypes.c_void_p]*2)
        fn.restype=ctypes.c_int
        _LIBRARIES[key]=(library,fn)
    return _LIBRARIES[key][1]


def native_arc_batch_metrics(arcs,contact_distance,robot_radius=.5,domain_side=800.,
                             continuity_tolerance=1e-8,library_path=None):
    """Return the same metric fields as arc_batch_metrics, plus native timing."""
    started=time.perf_counter(); arcs=list(arcs)
    if not arcs: raise ValueError('At least one arc required')
    if not np.isfinite([contact_distance,robot_radius,domain_side,continuity_tolerance]).all():
        raise ValueError('Finite geometry/tolerance required')
    if contact_distance<=0 or robot_radius<0 or domain_side<=2*robot_radius or continuity_tolerance<0:
        raise ValueError('Invalid geometry/tolerance')
    p=np.ascontiguousarray(np.stack([arc[0] for arc in arcs]),dtype=np.float64)
    v=np.ascontiguousarray(np.stack([arc[1] for arc in arcs]),dtype=np.float64)
    a=np.ascontiguousarray(np.stack([arc[2] for arc in arcs]),dtype=np.float64)
    duration=np.ascontiguousarray([arc[3] for arc in arcs],dtype=np.float64)
    if p.shape!=v.shape or p.shape!=a.shape or p.ndim!=3 or p.shape[2]!=2 or p.shape[1]<1:
        raise ValueError('Matching nonempty N-by-2 arc arrays required')
    if not np.isfinite([p,v,a]).all() or not np.isfinite(duration).all() or (duration<=0).any():
        raise ValueError('Finite states and positive durations required')
    h=duration[:,None,None]; end=p+h*v+.5*h*h*a; end_velocity=v+h*a
    if len(arcs)>1:
        mismatch=max(float(abs(end[:-1]-p[1:]).max()),float(abs(end_velocity[:-1]-v[1:]).max()))
        if mismatch>continuity_tolerance: raise ValueError(f'Noncontinuous arc sequence: {mismatch}')
    speed_bound=float(max(np.linalg.norm(v,axis=2).max(),np.linalg.norm(end_velocity,axis=2).max()))
    total_duration=float(duration.sum()); robots=p.shape[1]
    if robots>=2:
        tree=cKDTree(p[0]); initial_min=float(tree.query(p[0],k=2)[0][:,1].min())
        cutoff=max(initial_min,contact_distance)+2*speed_bound*total_duration+1e-8
        pairs=tree.query_pairs(cutoff,output_type='ndarray')
    else:
        initial_min=float('inf'); pairs=np.empty((0,2),dtype=np.int64)
    pairs=np.ascontiguousarray(pairs,dtype=np.int64)
    flags=np.zeros(len(pairs),dtype=np.uint8); output=np.empty(7,dtype=np.float64)
    arrays=[p,v,a,duration,pairs]
    code=_kernel(library_path)(len(arcs),robots,len(pairs),*[ctypes.c_void_p(x.ctypes.data) for x in arrays],
            contact_distance,robot_radius,domain_side,ctypes.c_void_p(output.ctypes.data),ctypes.c_void_p(flags.ctypes.data))
    if code!=0: raise RuntimeError(f'Native swept checker error: {code}')
    contacts=pairs[flags!=0]
    if len(contacts): contacts=contacts[np.lexsort((contacts[:,1],contacts[:,0]))]
    return dict(minimum_separation_m=min(initial_min,float(output[0])),contact_pairs=contacts,
                minimum_boundary_margin_m=float(output[1]),maximum_acceleration_m_s2=float(output[2]),
                maximum_speed_m_s=float(output[3]),travel_m=float(output[4]),candidate_pairs=len(pairs),
                arcs=len(arcs),duration_s=total_duration,native_seconds=float(output[6]),
                roots_checked=int(output[5]),seconds=time.perf_counter()-started)

