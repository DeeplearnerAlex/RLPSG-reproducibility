"""Pure motion helpers shared by the lower-level controller.

Extracted from the project implementation without changing their arithmetic.
The control interval is 0.1 seconds; no experiment driver is imported.
"""
import numpy as np

DT = .1


def sensed_repulsion(frame, population, reached, sensing_radius=3.):
    pairs=np.asarray(frame.sensed_pairs,dtype=np.int64).reshape(-1,2)
    force=np.zeros((population,2))
    if not len(pairs):return force
    relative=np.asarray(frame.sensed_relative_positions).copy()
    ages=np.asarray(frame.sensed_pair_ages)
    relative+=(frame.v_hat[pairs[:,1]]-frame.v_hat[pairs[:,0]])*ages[:,None]
    unique,inverse=np.unique(pairs,axis=0,return_inverse=True)
    relative_sum=np.zeros((len(unique),2));np.add.at(relative_sum,inverse,relative)
    relative=relative_sum/np.bincount(inverse)[:,None]
    distance=np.linalg.norm(relative,axis=1)
    active=(distance>1e-8)&(distance<sensing_radius)
    i,j=unique[active].T;d=distance[active]
    base=-relative[active]*((1/d-1/sensing_radius)/d**3)[:,None]
    strength=np.where(reached,5.,50.)
    np.add.at(force,i,base*strength[i,None]);np.add.at(force,j,-base*strength[j,None])
    return force


def bins_and_counts(points):
    valid = np.all((points >= 0) & (points < 800), axis=1)
    bins = np.full(len(points), -1, dtype=int)
    xy = np.floor(points[valid]/100).astype(int)
    bins[valid] = xy[:, 0] + 8*xy[:, 1]
    return bins, np.bincount(bins[valid], minlength=64), int((~valid).sum())


def clip_polygon(poly, normal, bound):
    if len(poly) == 0:
        return poly
    out = []
    prev = poly[-1]
    dp = float(prev @ normal-bound)
    for cur in poly:
        dc = float(cur @ normal-bound)
        if (dp <= 1e-10) != (dc <= 1e-10):
            out.append(prev + (cur-prev)*dp/(dp-dc))
        if dc <= 1e-10:
            out.append(cur)
        prev, dp = cur, dc
    return np.array(out).reshape(-1, 2)

