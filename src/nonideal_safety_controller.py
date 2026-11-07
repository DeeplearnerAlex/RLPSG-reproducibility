"""Estimate-only conservative EMPIRICAL command filter for validation studies.

No true positions/velocities, true-state candidate screening, or physical audit
enters this controller. Bounds are assumed inputs, not calibrated measurements.
The exact-state separation proof is NOT claimed for noisy/stale estimates. Pair,
wall and circular-obstacle constraints use different uncertainty radii at the
predicted execution endpoint and damped-braking reference point. A returned
emergency brake with estimated_feasible=False is NOT a safety certificate.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from reduced_velocity_dynamics import norm_clip
from relative_braking_filter import backup_parameters


def _bounds(value,n,name):
    result=np.broadcast_to(np.asarray(value,dtype=float),(n,)).copy()
    if not np.isfinite(result).all() or (result<0).any():raise ValueError(f'{name} must be finite and nonnegative')
    return result


def _circles(obstacles):
    value=np.asarray([] if obstacles is None else obstacles,dtype=float)
    if not value.size:return np.empty((0,3),dtype=float)
    if value.ndim!=2 or value.shape[1]!=3 or not np.isfinite(value).all() or (value[:,2]<=0).any():
        raise ValueError('obstacles must be a finite M-by-3 [center_x,center_y,radius] array')
    return value.copy()


def obstacle_aware_reference(p_hat,nominal,obstacles,radius=.5,clearance=.05,
                             influence_distance=8.,speed_cap=1.,chirality=1.):
    """Estimate-only circle repulsion/sliding reference, shared by both baselines.

    This explicit nominal-planning addition is not an unchanged VOAPF reference.
    Tangential sliding avoids an exactly head-on purely radial APF equilibrium.
    """
    p=np.asarray(p_hat,dtype=float);reference=np.asarray(nominal,dtype=float).copy()
    if p.shape!=reference.shape or p.ndim!=2 or p.shape[1]!=2:raise ValueError('Expected matching N-by-2 arrays')
    if influence_distance<=0:raise ValueError('Positive obstacle influence required')
    for x,y,obstacle_radius in _circles(obstacles):
        relative=p-np.array([x,y]);distance=np.linalg.norm(relative,axis=1)
        normal=relative/np.maximum(distance[:,None],1e-12)
        normal[distance<1e-12]=[1.,0.]
        gap=distance-(obstacle_radius+radius+clearance)
        active=gap<influence_distance
        if not active.any():continue
        weight=np.clip((influence_distance-gap)/influence_distance,0.,1.)
        inward=np.minimum(np.sum(reference*normal,axis=1),0.)
        tangent=np.column_stack((-normal[:,1],normal[:,0]))
        alignment=np.sum(reference*tangent,axis=1)
        direction=np.where(abs(alignment)>1e-10,np.sign(alignment),np.sign(chirality) or 1.)
        adjustment=(-weight*inward+.15*speed_cap*weight)[:,None]*normal
        adjustment+=((-.8*weight*inward+.15*speed_cap*weight)*direction)[:,None]*tangent
        reference[active]+=adjustment[active]
    return norm_clip(reference,speed_cap)


def _polygon_projection(desired,normal,limits):
    """Vectorized exact 2D polygon enumeration, with per-robot slew radii."""
    result=desired.copy();indices=np.flatnonzero(np.any(desired@normal.T>limits+1e-11,axis=1))
    if not len(indices):return result
    x=desired[indices];b=limits[indices]
    excess=x@normal.T-b;faces=x[:,None]-excess[:,:,None]*normal[None]
    first,second=np.triu_indices(len(normal),1)
    determinant=normal[first,0]*normal[second,1]-normal[first,1]*normal[second,0]
    keep=abs(determinant)>1e-12;first,second,determinant=first[keep],second[keep],determinant[keep]
    vx=(b[:,first]*normal[second,1]-normal[first,1]*b[:,second])/determinant
    vy=(normal[first,0]*b[:,second]-b[:,first]*normal[second,0])/determinant
    candidates=np.concatenate((faces,np.stack((vx,vy),axis=2)),axis=1)
    feasible=np.all(np.einsum('nkd,md->nkm',candidates,normal)<=b[:,None]+1e-10,axis=2)
    objective=np.where(feasible,np.sum((candidates-x[:,None])**2,axis=2),np.inf)
    selected=np.argmin(objective,axis=1)
    if not np.isfinite(objective[np.arange(len(indices)),selected]).all():raise ValueError('Empty conservative actuator polygon')
    result[indices]=candidates[np.arange(len(indices)),selected]
    return result


def _violation(matrix,lower,upper,candidate):
    if not np.isfinite(candidate).all():return float('inf')
    value=matrix@candidate.ravel()
    return float(max(0.,np.max(lower-value),np.max(value-upper)))


def _arc_circle_distances(points,velocity,acceleration,h,center):
    r=points-center;w=velocity;b=.5*acceleration
    end=r+h*w+h*h*b;squared=np.minimum(np.sum(r*r,axis=1),np.sum(end*end,axis=1))
    c0=np.sum(r*w,axis=1);c1=np.sum(w*w+2*r*b,axis=1)
    c2=3*np.sum(w*b,axis=1);c3=2*np.sum(b*b,axis=1)
    possible=(abs(c0)<=abs(c1)*h+abs(c2)*h*h+abs(c3)*h**3+1e-20)&((abs(c1)+abs(c2)+abs(c3))>0)
    for index in np.flatnonzero(possible):
        for root in np.roots(np.trim_zeros([c3[index],c2[index],c1[index],c0[index]],'f')):
            if abs(root.imag)<1e-9 and 0<root.real<h:
                t=float(root.real);value=r[index]+t*w[index]+t*t*b[index]
                squared[index]=min(squared[index],float(value@value))
    return np.sqrt(squared)


def _estimated_interval_clear(p,v,command,model,duration,tube,radius,safe,side,circles,clearance):
    """Predict ONLY estimated states; exact parabolic geometry, no hidden truth."""
    from accelerated_arc_batch_metrics import arc_batch_metrics
    _,_,arcs=model.advance(p,v,command,duration)
    largest=float(tube.max());distance=safe+2*largest
    metrics=arc_batch_metrics(arcs,distance,robot_radius=radius+largest,domain_side=side)
    if metrics['minimum_separation_m']<distance-1e-8 or metrics['minimum_boundary_margin_m']<-1e-8:return False
    for x,y,obstacle_radius in circles:
        required=obstacle_radius+radius+clearance+tube
        for point,velocity,acceleration,h in arcs:
            distances=_arc_circle_distances(point,velocity,acceleration,h,np.array([x,y]))
            if (distances<required-1e-8).any():return False
    return True


def filter_estimated(p_hat,v_hat,nominal,model,position_error=0.,velocity_error=0.,obstacles=None,
                     age_s=0.,bounds_are_current=True,duration=.1,radius=.5,safe_distance=1.05,
                     domain_side=800.,obstacle_clearance=.05,projection_state=None,
                     native_sweeps=100,restoration_sweeps=800,componentwise=True,
                     obstacle_reference=True,obstacle_influence=8.,model_acceleration_error=0.,
                     maximum_position_error=.25,maximum_velocity_error=.25,max_information_age_s=None,
                     weighted_tube_normals=False):
    """Return command, JSON diagnostics using ESTIMATES ONLY.

    position_error/velocity_error are CURRENT radius bounds by default. If not
    current, p_hat/v_hat must already be extrapolated estimates; only their bounds
    are grown by ep+age*ev+.5*amax*age² and ev+amax*age. The online controller never
    accesses the hidden state used by the runner's independent physical audit.
    """
    started=time.perf_counter()
    p,v,reference=(np.asarray(a,dtype=float).copy() for a in [p_hat,v_hat,nominal])
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or v.shape!=p.shape or reference.shape!=p.shape or not np.isfinite([p,v,reference]).all():
        raise ValueError('Finite matching N-by-2 estimated states/reference required')
    n=len(p);circles=_circles(obstacles)
    original_nominal=reference.copy()
    ep=_bounds(position_error,n,'position_error');ev=_bounds(velocity_error,n,'velocity_error');age=_bounds(age_s,n,'age_s')
    if not np.isfinite([duration,radius,safe_distance,domain_side,obstacle_clearance,model_acceleration_error]).all() or min(duration,radius,safe_distance,domain_side)<=0 or min(obstacle_clearance,model_acceleration_error)<0:
        raise ValueError('Invalid geometry/model uncertainty')
    if not bounds_are_current:
        ep=ep+age*ev+.5*model.acceleration_cap_m_s2*age**2
        ev=ev+model.acceleration_cap_m_s2*age
    v=norm_clip(v,model.speed_cap_m_s)
    param=backup_parameters(model,duration,8);A,B,q,K=(param[k] for k in ['A','B','q','K'])
    zv,zc=A+K*q,B+K*(1-q)
    # Emergency braking is an estimated, bounded command, not an instantaneous
    # true-velocity reset or a hidden true-state safety oracle.
    emergency=param['damping']*v
    info=dict(method='estimate-only uncertainty-tube empirical command filter',used_true_state=False,
        deterministic_safety_certificate=False,estimated_feasible=False,emergency_brake=False,
        maximum_position_error_m=float(ep.max()),maximum_velocity_error_m_s=float(ev.max()),
        maximum_information_age_s=float(age.max()),bounds_are_current=bool(bounds_are_current),
        modified_robots=0,emergency=False,estimated_full_interval_checks=0,
        components=0,failed_components=[],native_calls=0,restoration_calls=0,osqp_calls=0,
        scipy_calls=0,pair_constraints=0,obstacle_constraints=0,current_tube_domain_warning=False,
        obstacle_reference_modified=bool(obstacle_reference and len(circles)),
        weighted_tube_normals=bool(weighted_tube_normals),
        estimated_prediction_verification='P/Z support constraints; full estimated parabolic subarcs on current-domain-warning components',
        scope='Assumed-bound, estimate-model conservative experiment; not calibrated fidelity or a deterministic nonideal safety proof')
    def early(reason):
        info.update(reason=reason,emergency_brake=True,emergency=True,
            modified_robots=int((np.linalg.norm(emergency-original_nominal,axis=1)>1e-6).sum()),seconds=time.perf_counter()-started)
        return emergency,info
    if ep.max()>maximum_position_error or ev.max()>maximum_velocity_error:
        return early('uncertainty_bound_exceeds_configured_budget')
    if max_information_age_s is not None and age.max()>max_information_age_s:
        return early('information_age_exceeds_configured_budget')
    reach=model.tau_s*model.acceleration_cap_m_s2-ev
    if (reach<=0).any():return early('velocity_uncertainty_exhausts_actuator_reachability')
    if obstacle_reference:
        reference=obstacle_aware_reference(p,reference,circles,radius,obstacle_clearance,
                                          obstacle_influence,model.speed_cap_m_s)
    rho_p=ep+A*ev+.5*duration**2*model_acceleration_error
    rho_z=ep+zv*ev+.5*(duration+K)**2*model_acceleration_error
    rho_stop=ep+K*ev+.5*K*K*model_acceleration_error
    cutoff=safe_distance+2*K*model.speed_cap_m_s+2*duration*model.speed_cap_m_s+2*max(rho_p.max(),rho_z.max())+1e-8
    pairs=cKDTree(p).query_pairs(cutoff,output_type='ndarray')
    blocks=[];lows=[];highs=[];keys=[];initial_margins=[];warning_robots=np.zeros(n,dtype=bool)
    if len(pairs):
        i,j=pairs.T;r=p[i]-p[j];u=v[i]-v[j]
        if weighted_tube_normals:
            from nonideal_weighted_tube_geometry import tapered_segment_support
            error_slope=ev+.5*K*model_acceleration_error
            normal,beta,_=tapered_segment_support(r,u,error_slope[i]+error_slope[j],K,ep[i]+ep[j],safe_distance)
        else:
            beta=np.clip(-np.sum(r*u,axis=1)/np.maximum(np.sum(u*u,axis=1),1e-30),0.,K)
            closest=r+beta[:,None]*u;normal=closest/np.maximum(np.linalg.norm(closest,axis=1,keepdims=True),1e-30)
        bp=(safe_distance+rho_p[i]+rho_p[j]-np.sum(normal*(r+A*u),axis=1))/B
        bz=(safe_distance+rho_z[i]+rho_z[j]-np.sum(normal*(r+zv*u),axis=1))/zc
        blocks.append(sparse.csr_matrix((np.column_stack((normal,-normal)).ravel(),
            (np.repeat(np.arange(len(pairs)),4),np.column_stack((2*i,2*i+1,2*j,2*j+1)).ravel())),shape=(len(pairs),2*n)))
        lows.append(np.maximum(bp,bz));highs.append(np.full(len(pairs),np.inf));keys.append(i*n+j)
        current_margin=np.sum(normal*r,axis=1)-safe_distance-ep[i]-ep[j]
        stopping_margin=np.sum(normal*(r+K*u),axis=1)-safe_distance-rho_stop[i]-rho_stop[j]
        initial_margins.extend([float(current_margin.min()),float(stopping_margin.min())])
        invalid=(current_margin<-1e-8)|(stopping_margin<-1e-8);warning_robots[i[invalid]]=True;warning_robots[j[invalid]]=True
    info['pair_constraints']=len(pairs)
    angle=np.arange(8)*2*np.pi/8;normals=np.column_stack((np.cos(angle),np.sin(angle)));rho=np.cos(np.pi/8)
    limits=np.minimum(model.speed_cap_m_s*rho,v@normals.T+reach[:,None]*rho)
    blocks.append(sparse.kron(sparse.eye(n,format='csr'),sparse.csr_matrix(normals),format='csr'))
    lows.append(np.full(n*8,-np.inf));highs.append(limits.ravel());keys.append(n*n+np.arange(n*8))
    low=np.maximum((radius+rho_p[:,None]-p-A*v)/B,(radius+rho_z[:,None]-p-zv*v)/zc).ravel()
    high=np.minimum((domain_side-radius-rho_p[:,None]-p-A*v)/B,(domain_side-radius-rho_z[:,None]-p-zv*v)/zc).ravel()
    blocks.append(sparse.eye(2*n,format='csr'));lows.append(low);highs.append(high);keys.append(n*n+8*n+np.arange(2*n))
    wall_margins=[p-radius-ep[:,None],domain_side-radius-p-ep[:,None],
        p+K*v-radius-rho_stop[:,None],domain_side-radius-p-K*v-rho_stop[:,None]]
    initial_margins.extend(float(value.min()) for value in wall_margins)
    for value in wall_margins:warning_robots|=np.min(value,axis=1)<-1e-8
    if len(circles):
        robot_index=[];obstacle_index=[];ob_normals=[];ob_lower=[]
        for obstacle,(x,y,ob_radius) in enumerate(circles):
            r=p-np.array([x,y])
            if weighted_tube_normals:
                from nonideal_weighted_tube_geometry import tapered_segment_support
                weighted_normal,beta,_=tapered_segment_support(r,v,ev+.5*K*model_acceleration_error,K,ep,
                    radius+ob_radius+obstacle_clearance)
                nearest=r+beta[:,None]*v;distance=np.linalg.norm(nearest,axis=1)
            else:
                beta=np.clip(-np.sum(r*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-30),0.,K)
                nearest=r+beta[:,None]*v;distance=np.linalg.norm(nearest,axis=1)
            d=radius+ob_radius+obstacle_clearance
            active=np.flatnonzero(np.linalg.norm(r,axis=1)<=d+K*model.speed_cap_m_s+duration*model.speed_cap_m_s+max(rho_p.max(),rho_z.max())+1e-8)
            if not len(active):continue
            normal=weighted_normal[active] if weighted_tube_normals else nearest[active]/np.maximum(distance[active,None],1e-30)
            normal[distance[active]<1e-12]=[1.,0.]
            bp=(d+rho_p[active]-np.sum(normal*(r[active]+A*v[active]),axis=1))/B
            bz=(d+rho_z[active]-np.sum(normal*(r[active]+zv*v[active]),axis=1))/zc
            robot_index.extend(active.tolist());obstacle_index.extend([obstacle]*len(active))
            ob_normals.extend(normal.tolist());ob_lower.extend(np.maximum(bp,bz).tolist())
            current_margin=np.sum(normal*r[active],axis=1)-d-ep[active]
            stopping_margin=np.sum(normal*(r[active]+K*v[active]),axis=1)-d-rho_stop[active]
            initial_margins.extend([float(current_margin.min()),float(stopping_margin.min())])
            warning_robots[active[(current_margin<-1e-8)|(stopping_margin<-1e-8)]]=True
        count=len(robot_index)
        if count:
            ids=np.asarray(robot_index);normal=np.asarray(ob_normals)
            blocks.append(sparse.csr_matrix((normal.ravel(),(np.repeat(np.arange(count),2),
                np.column_stack((2*ids,2*ids+1)).ravel())),shape=(count,2*n)))
            lows.append(np.array(ob_lower));highs.append(np.full(count,np.inf))
            keys.append(n*n+10*n+ids*len(circles)+np.asarray(obstacle_index))
        info['obstacle_constraints']=count
    matrix=sparse.vstack(blocks,format='csr');lower=np.concatenate(lows);upper=np.concatenate(highs);row_keys=np.concatenate(keys).astype(np.int64)
    info['estimated_current_support_margin_m']=min(initial_margins)
    info['current_tube_domain_warning']=bool(min(initial_margins)<-1e-8)
    info.update(rows=matrix.shape[0],nonzeros=matrix.nnz,rho_execution_max_m=float(rho_p.max()),rho_braking_max_m=float(rho_z.max()))
    try:candidate=_polygon_projection(reference,normals,limits)
    except ValueError:return early('empty_conservative_actuator_polygon')
    state={} if projection_state is None else projection_state
    old={int(k):float(d) for k,d in zip(state.get('row_keys',[]),state.get('dual',[]))} if state.get('population')==n else {}
    if componentwise:
        graph=sparse.csr_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n))
        component_count,labels=connected_components(graph,directed=False)
    else:component_count,labels=1,np.zeros(n,dtype=int)
    info['components']=int(component_count);row_components=labels[matrix.indices[matrix.indptr[:-1]]//2]
    if not np.array_equal(labels[matrix.indices//2],np.repeat(row_components,np.diff(matrix.indptr))):
        raise RuntimeError('A coupled inequality crossed the component partition')
    result=candidate.copy();new_dual=np.zeros(len(row_keys));details=[]
    values=matrix@candidate.ravel()
    failed_rows=(lower-values>2e-8)|(values-upper>2e-8)
    warning_labels=np.unique(labels[warning_robots])
    affected=np.union1d(np.unique(row_components[failed_rows]),warning_labels)
    info['directly_accepted_components']=int(component_count-len(affected))
    info['maximum_component_population']=int(np.bincount(labels).max())
    for label in affected:
        ids=np.flatnonzero(labels==label);columns=np.column_stack((2*ids,2*ids+1)).ravel();rows=np.flatnonzero(row_components==label)
        mat=matrix[rows][:,columns];lo=lower[rows];hi=upper[rows];desired=reference[ids]
        part=candidate[ids];dual=None;method='independent_actuator_projection';reason=None
        def acceptable(command):
            linear_ok=(_violation(mat,lo,hi,command)<=2e-8 and
                np.linalg.norm(command,axis=1).max()<=model.speed_cap_m_s+1e-8 and
                np.max(np.linalg.norm(command-v[ids],axis=1)+ev[ids])<=model.tau_s*model.acceleration_cap_m_s2+1e-8)
            if not linear_ok:return False
            if label in warning_labels:
                info['estimated_full_interval_checks']+=1
                return _estimated_interval_clear(p[ids],v[ids],command,model,duration,rho_p[ids],radius,
                    safe_distance,domain_side,circles,obstacle_clearance)
            return True
        ok=acceptable(part)
        if not ok:
            try:
                from warm_command_projection import project_command_polytope_warm
                mapped=np.array([old.get(int(key),0.) for key in row_keys[rows]])
                mapped[(np.isposinf(hi)&(mapped>0))|(np.isneginf(lo)&(mapped<0))]=0.
                part,dual,native=project_command_polytope_warm(desired,mat,lo,hi,dual=mapped,max_sweeps=native_sweeps)
                if not np.isfinite(part).all():raise FloatingPointError('Nonfinite native candidate')
                info['native_calls']+=1;method='finite_native_candidate';ok=acceptable(part)
                if not ok:
                    restored,repair_dual,repair=project_command_polytope_warm(part,mat,lo,hi,max_sweeps=restoration_sweeps)
                    info['restoration_calls']+=1;ok=acceptable(restored)
                    if ok:part=restored;dual=dual+repair_dual;method='candidate_centered_feasibility_repair'
            except Exception as error:reason=type(error).__name__;part=candidate[ids].copy();dual=None
        if not ok:
            try:
                import osqp
                problem=osqp.OSQP();problem.setup(P=sparse.eye(len(columns),format='csc'),q=-desired.ravel(),A=mat.tocsc(),
                    l=lo,u=hi,verbose=False,eps_abs=1e-5,eps_rel=1e-5,rho=1.,max_iter=1000,polishing=True)
                problem.warm_start(x=emergency[ids].ravel());solution=problem.solve(raise_error=False);info['osqp_calls']+=1
                if solution.x is not None and np.isfinite(solution.x).all():
                    trial=solution.x.reshape(len(ids),2)
                    if acceptable(trial):part=trial;dual=solution.y;ok=True;method='osqp_estimated_feasible'
                reason=solution.info.status
            except ImportError:
                if len(columns)<=32:
                    from scipy.optimize import minimize,LinearConstraint
                    try:
                        solution=minimize(lambda x:.5*np.sum((x-desired.ravel())**2),part.ravel(),jac=lambda x:x-desired.ravel(),
                            method='SLSQP',constraints=[LinearConstraint(mat.toarray(),lo,hi)],options=dict(ftol=1e-11,maxiter=200))
                        info['scipy_calls']+=1
                        trial=solution.x.reshape(len(ids),2)
                        if acceptable(trial):part=trial;dual=None;ok=True;method='scipy_estimated_feasible'
                        reason=str(solution.message)
                    except Exception as error:reason=type(error).__name__
            except Exception as error:reason=type(error).__name__
        if ok:
            result[ids]=part
            if dual is not None and np.isfinite(dual).all():new_dual[rows]=dual
        else:
            result[ids]=emergency[ids];info['failed_components'].append(ids.tolist())
        details.append(dict(population=len(ids),estimated_feasible=bool(ok),method=method,solver_reason=reason))
    state.update(population=n,row_keys=row_keys.copy(),dual=new_dual.copy())
    info.update(estimated_feasible=not info['failed_components'],emergency_brake=bool(info['failed_components']),
        emergency=bool(info['failed_components']),modified_robots=int((np.linalg.norm(result-original_nominal,axis=1)>1e-6).sum()),
        component_details=details,estimated_constraint_residual=_violation(matrix,lower,upper,result),
        reason='estimated_constraints_satisfied' if not info['failed_components'] else 'infeasible_components_use_estimated_emergency_brake',
        seconds=time.perf_counter()-started)
    return result,info

