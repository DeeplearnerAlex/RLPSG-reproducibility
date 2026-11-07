"""Optional identity-level terminal allocator; no policy or physical integration.

Existing positive-target goals are frozen. Only originally zero-target identities
receive routes through zero-target bins to reserved positive-target vacancies.
Diagonal edges require BOTH orthogonal side bins to have zero target density.
This is a new terminal rule, not an unchanged PPO/position-adjustment evaluation.
"""
from __future__ import annotations

import copy
import heapq
import json
import math
import numpy as np

from reduced_velocity_dynamics import finite_population_hd_floor

DIAGONAL_RULE = 'both_orthogonal_side_bins_must_have_zero_target'


def _counts_phi(counts,phi):
    c=np.asarray(counts);p=np.asarray(phi,dtype=float)
    if c.ndim!=1 or p.shape!=c.shape or not len(c) or not np.isfinite(c).all() or (c<0).any() or (c!=np.floor(c)).any():
        raise ValueError('counts must be a nonnegative integer vector matching phi')
    if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:
        raise ValueError('phi must be finite, nonnegative, and have positive mass')
    return c.astype(np.int64),p/p.sum()


def zero_target_population(counts,phi):
    c,p=_counts_phi(counts,phi)
    return int(c[p==0].sum())


def strict_success(counts,phi,threshold=.05):
    c,p=_counts_phi(counts,phi);n=int(c.sum())
    if n<1 or not np.isfinite(threshold) or threshold<=0:return False
    error=float(np.linalg.norm(np.sqrt(c/n)-np.sqrt(p))/np.sqrt(2))
    return bool(error<threshold and zero_target_population(c,p)==0)


class TerminalZeroCleanupAllocator:
    """Fixed-population per-ID next-hop assignment, with JSON continuation state.

    API: allocate(points, velocity=None, counts=None, allocation_index=None).
    IDs are stable array indices 0..population-1; births/deaths/reordering are not
    supported during this terminal mode. A blocked ID is assigned its actual bin
    to hold, and is explicitly reported; it is never sent through a full target.
    """
    def __init__(self,target,population,domain_side=800.,grid=None):
        if not isinstance(population,(int,np.integer)) or isinstance(population,(bool,np.bool_)) or population<1:
            raise ValueError('population must be a positive integer')
        phi=np.asarray(target,dtype=float)
        if phi.ndim!=1 or not len(phi) or not np.isfinite(phi).all() or (phi<0).any() or phi.sum()<=0:
            raise ValueError('Invalid target density')
        inferred=int(round(np.sqrt(len(phi))))
        grid=inferred if grid is None else grid
        if not isinstance(grid,(int,np.integer)) or grid<1 or grid*grid!=len(phi):
            raise ValueError('target must describe a square grid')
        if not np.isfinite(domain_side) or domain_side<=0:raise ValueError('Positive finite domain side required')
        self.target=phi/phi.sum();self.population=int(population);self.grid=int(grid)
        self.domain_side=float(domain_side);self.bin_side=self.domain_side/self.grid
        self.best_hd,self.quota=finite_population_hd_floor(self.target,self.population)
        self.target.flags.writeable=False;self.quota.flags.writeable=False
        self._records=None;self._last_bins=None;self._last_counts=None;self._calls=0
        self.events=[];self.last_report={}

    def _bins(self,points):
        p=np.asarray(points,dtype=float)
        if p.shape!=(self.population,2) or not np.isfinite(p).all() or ((p<0)|(p>=self.domain_side)).any():
            raise ValueError('Expected finite in-domain N-by-2 actual positions')
        xy=np.floor(p/self.bin_side).astype(int)
        return xy[:,0]+self.grid*xy[:,1]

    def _event(self,kind,allocation,**values):
        self.events.append(dict(type=kind,call=self._calls,allocation_index=allocation,**values))

    @property
    def frozen_goal_bins(self):
        """Fresh N-vector: locked positive goal, or -1 for unfinished cleanup."""
        if self._records is None:return np.full(self.population,-1,dtype=np.int64)
        return np.array([r['locked_bin'] if r['status'] in ['frozen','delivered'] else -1
                         for r in self._records],dtype=np.int64)

    def _slot(self,destination,counts,reserved):
        effective=int(counts[destination]+reserved[destination])
        return bool(self.target[destination]>0 and effective<int(self.quota[destination]) and
                    np.float32(effective/self.population)<np.float32(self.target[destination]))

    def _neighbors(self,node):
        x,y=node%self.grid,node//self.grid
        result=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if not (dx or dy):continue
                xx,yy=x+dx,y+dy
                if not (0<=xx<self.grid and 0<=yy<self.grid):continue
                if dx and dy:
                    if self.target[x+dx+self.grid*y]>0 or self.target[x+self.grid*(y+dy)]>0:continue
                result.append((xx+self.grid*yy,self.bin_side*(math.sqrt(2) if dx and dy else 1.)))
        return sorted(result)

    def _route(self,source,counts,reserved):
        available=[i for i in range(len(self.target)) if self._slot(i,counts,reserved)]
        if not available:return None,'no_admissible_target_capacity'
        available=set(available);best={source:(0.,(source,))};queue=[(0.,(source,),source)];found={}
        while queue:
            distance,path,node=heapq.heappop(queue)
            if (distance,path)!=best[node]:continue
            for neighbor,cost in self._neighbors(node):
                length=distance+cost;next_path=path+(neighbor,)
                if self.target[neighbor]>0:
                    if neighbor in available:
                        old=found.get(neighbor)
                        if old is None or length<old[0]-1e-9 or (abs(length-old[0])<=1e-9 and next_path<old[1]):
                            found[neighbor]=(length,next_path)
                    continue  # A positive target is NEVER an intermediate node.
                old=best.get(neighbor)
                if old is None or length<old[0]-1e-9 or (abs(length-old[0])<=1e-9 and next_path<old[1]):
                    best[neighbor]=(length,next_path);heapq.heappush(queue,(length,next_path,neighbor))
        if not found:return None,'no_legal_zero_target_route'
        shortest=min(value[0] for value in found.values())
        candidates=[destination for destination,value in found.items() if abs(value[0]-shortest)<=1e-9]
        destination=min(candidates,key=lambda i:(-int(self.quota[i]-counts[i]-reserved[i]),i,found[i][1]))
        length,path=found[destination]
        return (list(path),float(length)),None

    def _pending(self,record,actual):
        record.update(status='pending',route=[],route_index=0,final_bin=None,reserved_bin=None,
                      next_hop=int(actual),blocked_reason=None)

    def allocate(self,points,velocity=None,counts=None,allocation_index=None):
        bins=self._bins(points);actual_counts=np.bincount(bins,minlength=len(self.target))
        if velocity is not None:
            velocity=np.asarray(velocity)
            if velocity.shape!=(self.population,2) or not np.isfinite(velocity).all():raise ValueError('Invalid actual velocity')
        if counts is not None:
            supplied,_=_counts_phi(counts,self.target)
            if not np.array_equal(supplied,actual_counts):raise ValueError('Supplied counts disagree with actual positions')
        if allocation_index is not None:
            if not isinstance(allocation_index,(int,np.integer)):raise ValueError('allocation_index must be an integer or None')
            allocation_index=int(allocation_index)
        self._calls+=1;event_start=len(self.events)
        if self._records is None:
            self._records=[]
            for identity,actual in enumerate(bins):
                positive=bool(self.target[actual]>0)
                self._records.append(dict(id=identity,kind='initial_positive' if positive else 'cleanup',
                    status='frozen' if positive else 'pending',locked_bin=int(actual) if positive else None,
                    route=[],route_index=0,final_bin=None,reserved_bin=None,next_hop=int(actual),
                    last_observed_bin=int(actual),blocked_reason=None))
            self._event('initialized',allocation_index,frozen_positive_ids=[r['id'] for r in self._records if r['kind']=='initial_positive'],
                cleanup_ids=[r['id'] for r in self._records if r['kind']=='cleanup'],diagonal_rule=DIAGONAL_RULE,
                quota_method='finite_population_hd_floor',receiving_precision='float32')
        for record,observed in zip(self._records,bins):
            observed=int(observed);previous=record['last_observed_bin']
            if record['status'] in ['frozen','delivered']:
                if observed!=record['locked_bin'] and observed!=previous:
                    self._event('locked_goal_physical_deviation',allocation_index,id=record['id'],actual_bin=observed,locked_bin=record['locked_bin'])
            elif self.target[observed]>0:
                unexpected=observed!=record['final_bin']
                self._event('unexpected_positive_delivery' if unexpected else 'delivered',allocation_index,
                    id=record['id'],actual_bin=observed,planned_final_bin=record['final_bin'],
                    actual_count=int(actual_counts[observed]),quota=int(self.quota[observed]),
                    within_integer_quota=bool(actual_counts[observed]<=self.quota[observed]))
                delivered_route=[observed] if unexpected else record['route']
                record.update(status='delivered',locked_bin=observed,final_bin=observed,reserved_bin=None,
                    route=delivered_route,route_index=max(0,len(delivered_route)-1),next_hop=observed,blocked_reason=None)
            elif record['status']=='inflight':
                if observed==record['next_hop']:
                    record['route_index']+=1
                    record['next_hop']=record['route'][record['route_index']+1]
                    self._event('hop_arrived',allocation_index,id=record['id'],actual_bin=observed,next_hop=record['next_hop'])
                elif observed!=record['route'][record['route_index']]:
                    old_route=record['route'].copy();old_final=record['final_bin']
                    self._pending(record,observed)
                    self._event('zero_route_deviation_replan',allocation_index,id=record['id'],actual_bin=observed,
                        old_route=old_route,released_final_bin=old_final)
            elif observed!=previous:
                self._pending(record,observed)
                self._event('pending_zero_deviation_replan',allocation_index,id=record['id'],actual_bin=observed)
            record['last_observed_bin']=observed
        # Locked identities that drifted away still own their original slots.
        reserved=np.zeros(len(self.target),dtype=np.int64);reservations=[]
        for record,actual in zip(self._records,bins):
            if record['status'] in ['frozen','delivered'] and actual!=record['locked_bin']:
                destination=record['locked_bin'];reserved[destination]+=1
                reservations.append(dict(id=record['id'],destination_bin=destination,kind='locked_return'))
        for record in self._records:
            if record['status']!='inflight':continue
            destination=record['reserved_bin']
            if self._slot(destination,actual_counts,reserved):
                reserved[destination]+=1
                reservations.append(dict(id=record['id'],destination_bin=destination,kind='cleanup'))
            else:
                actual=int(bins[record['id']]);self._pending(record,actual)
                self._event('reservation_released_capacity_changed',allocation_index,id=record['id'],old_final_bin=destination)
        for record in self._records:
            if record['status'] not in ['pending','blocked']:continue
            source=int(bins[record['id']]);planned,reason=self._route(source,actual_counts,reserved)
            if planned is None:
                record.update(status='blocked',next_hop=source,blocked_reason=reason)
                self._event('blocked',allocation_index,id=record['id'],actual_bin=source,reason=reason)
                continue
            route,length=planned;destination=route[-1]
            record.update(status='inflight',route=route,route_index=0,final_bin=destination,
                reserved_bin=destination,next_hop=route[1],blocked_reason=None)
            reserved[destination]+=1
            reservations.append(dict(id=record['id'],destination_bin=destination,kind='cleanup'))
            self._event('route_assigned',allocation_index,id=record['id'],route=route.copy(),final_bin=destination,
                graph_path_length_m=length,diagonal_rule=DIAGONAL_RULE)
        goals=np.array([r['locked_bin'] if r['status'] in ['frozen','delivered'] else r['next_hop'] for r in self._records],dtype=np.int64)
        self._last_bins=bins.tolist();self._last_counts=actual_counts.tolist()
        error=float(np.linalg.norm(np.sqrt(actual_counts/self.population)-np.sqrt(self.target))/np.sqrt(2))
        self.last_report=dict(allocation_index=allocation_index,zero_target_population=zero_target_population(actual_counts,self.target),
            hd=error,strict_success=strict_success(actual_counts,self.target),quota=self.quota.tolist(),
            current_counts=actual_counts.tolist(),reserved_counts=reserved.tolist(),reservations=reservations,
            blocked=[dict(id=r['id'],bin=int(bins[r['id']]),reason=r['blocked_reason']) for r in self._records if r['status']=='blocked'],
            routes=[dict(id=r['id'],route=r['route'].copy(),route_index=r['route_index'],next_hop=r['next_hop'],
                final_bin=r['final_bin'],status=r['status']) for r in self._records if r['kind']=='cleanup'],
            frozen_goal_bins=self.frozen_goal_bins.tolist(),
            diagonal_rule=DIAGONAL_RULE,events_this_call=copy.deepcopy(self.events[event_start:]))
        return goals

    def state_dict(self):
        return copy.deepcopy(dict(version=1,target=self.target.tolist(),population=self.population,grid=self.grid,
            domain_side=self.domain_side,quota=self.quota.tolist(),diagonal_rule=DIAGONAL_RULE,calls=self._calls,
            records=self._records,last_bins=self._last_bins,last_counts=self._last_counts,
            events=self.events,last_report=self.last_report))

    def load_state_dict(self,state):
        try:s=json.loads(json.dumps(state,allow_nan=False))
        except (TypeError,ValueError) as exc:raise ValueError('State must be finite JSON data') from exc
        if (s.get('version')!=1 or s.get('population')!=self.population or s.get('grid')!=self.grid or
            s.get('domain_side')!=self.domain_side or s.get('diagonal_rule')!=DIAGONAL_RULE or
            not np.array_equal(s.get('target'),self.target) or not np.array_equal(s.get('quota'),self.quota)):
            raise ValueError('State configuration/target/quota mismatch')
        records=s.get('records');last_bins=s.get('last_bins');last_counts=s.get('last_counts')
        if not isinstance(s.get('calls'),int) or s['calls']<0 or not isinstance(s.get('events'),list):raise ValueError('Invalid state history')
        if records is not None:
            if not isinstance(records,list) or len(records)!=self.population or not isinstance(last_bins,list) or len(last_bins)!=self.population:
                raise ValueError('Invalid identity/state dimensions')
            if any(not isinstance(b,int) or not 0<=b<len(self.target) for b in last_bins):raise ValueError('Invalid last bins')
            actual=np.bincount(last_bins,minlength=len(self.target))
            if not np.array_equal(actual,last_counts):raise ValueError('Last counts disagree with last bins')
            reserved=np.zeros(len(self.target),dtype=int)
            for identity,r in enumerate(records):
                if r.get('id')!=identity or r.get('kind') not in ['initial_positive','cleanup'] or r.get('status') not in ['frozen','delivered','pending','blocked','inflight']:
                    raise ValueError('Invalid identity record')
                if r.get('last_observed_bin')!=last_bins[identity]:raise ValueError('Last observation/identity mismatch')
                if r['kind']=='cleanup' and r['status']=='frozen':raise ValueError('Cleanup identity cannot become an initial frozen identity')
                if r['status'] in ['frozen','delivered']:
                    b=r.get('locked_bin')
                    if not isinstance(b,int) or not 0<=b<len(self.target) or self.target[b]<=0:raise ValueError('Invalid locked positive bin')
                    if r['kind']=='initial_positive' and r['status']!='frozen':raise ValueError('Initial positive goal must remain frozen')
                    if r.get('reserved_bin') is not None or r.get('next_hop')!=b:raise ValueError('Invalid locked-goal reservation')
                    if last_bins[identity]!=b:reserved[b]+=1
                elif r['kind']!='cleanup' or self.target[last_bins[identity]]!=0:raise ValueError('Invalid unfinished cleanup identity')
                elif r['status'] in ['pending','blocked'] and (r.get('reserved_bin') is not None or r.get('final_bin') is not None
                        or r.get('route')!=[] or r.get('next_hop')!=last_bins[identity]):
                    raise ValueError('Unplanned/blocked identity contains an active reservation')
            for identity,r in enumerate(records):
                if r['status']!='inflight':continue
                route=r.get('route');index=r.get('route_index');destination=r.get('reserved_bin')
                if (not isinstance(route,list) or len(route)<2 or any(not isinstance(b,int) or not 0<=b<len(self.target) for b in route)
                    or not isinstance(index,int) or not 0<=index<len(route)-1 or r.get('next_hop')!=route[index+1]
                    or last_bins[identity]!=route[index] or destination!=route[-1] or r.get('final_bin')!=destination
                    or any(self.target[b]>0 for b in route[:-1]) or not self._slot(destination,actual,reserved)):
                    raise ValueError('Invalid route/reservation')
                for first,second in zip(route[:-1],route[1:]):
                    if second not in dict(self._neighbors(first)):raise ValueError('Illegal route/corner cutting')
                reserved[destination]+=1
        elif last_bins is not None or last_counts is not None or s['calls']!=0:raise ValueError('Invalid uninitialized state')
        self._records=records;self._last_bins=last_bins;self._last_counts=last_counts
        self._calls=s['calls'];self.events=s['events'];self.last_report=s.get('last_report',{})
        return self

