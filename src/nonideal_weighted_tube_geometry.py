"""Supporting normals for a segment with linearly growing uncertainty radius.

This fixes a geometric subproblem, not a nonideal closed-loop safety theorem.
For r+beta*u, minimize ||r+beta*u||-s-e_p-beta*e_v on [0,K]. At an interior
minimum n.u=e_v, so n supports BOTH endpoint balls with different radii. The
ordinary closest-centerline normal (e_v=0) need not have that property.
"""
import numpy as np


def tapered_segment_support(relative_position,relative_velocity,velocity_error,horizon,
                            position_error=0.,safe_distance=0.):
    """Return (unit_normal, beta, minimum_margin) for N independent segments.

    Inputs have shape (N,2), scalar or (N,) nonnegative errors, and scalar K>=0.
    In the already-unsafe zero-vector degeneracy a fixed normal is returned and
    the nonpositive minimum margin is retained; it is not made falsely safe.
    """
    r=np.asarray(relative_position,dtype=float);u=np.asarray(relative_velocity,dtype=float)
    if r.ndim!=2 or r.shape[1]!=2 or u.shape!=r.shape or not np.isfinite([r,u]).all():
        raise ValueError('Finite matching N-by-2 relative arrays required')
    n=len(r)
    e=np.broadcast_to(np.asarray(velocity_error,dtype=float),(n,)).copy()
    ep=np.broadcast_to(np.asarray(position_error,dtype=float),(n,)).copy()
    s=np.broadcast_to(np.asarray(safe_distance,dtype=float),(n,)).copy()
    if not np.isfinite([horizon]).all() or horizon<0 or not all(np.isfinite(x).all() and (x>=0).all() for x in [e,ep,s]):
        raise ValueError('Finite nonnegative horizon, radius and error bounds required')
    uu=np.sum(u*u,axis=1);ru=np.sum(r*u,axis=1)
    beta=np.full(n,float(horizon))
    active=(uu>e*e)&(uu>0)
    if active.any():
        a=uu[active];b=ru[active];speed=np.sqrt(a);projection=b/a
        perpendicular=r[active]-projection[:,None]*u[active]
        # Equivalent to e*sqrt(max(A*C-B²,0))/sqrt(A-e²), but the
        # perpendicular-vector form avoids cancellation for near-collinear data.
        correction=(e[active]/speed)*(np.linalg.norm(perpendicular,axis=1)/np.sqrt(a-e[active]**2))
        beta[active]=np.clip(-projection+correction,0.,horizon)
    closest=r+beta[:,None]*u;length=np.linalg.norm(closest,axis=1)
    normal=np.divide(closest,length[:,None],out=np.zeros_like(closest),where=length[:,None]>1e-30)
    normal[length<=1e-30]=[1.,0.]
    margin=length-s-ep-beta*e
    return normal,beta,margin

