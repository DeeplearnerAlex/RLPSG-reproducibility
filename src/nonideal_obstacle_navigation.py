"""Estimated-position obstacle detours for the nonideal validation variant.

This changes velocity REFERENCES, never positions or goal-bin identities. It is
not a safety certificate: delayed/noisy observations, execution error and actual
continuous clearances are handled and audited by the separate execution layer.
All navigation tuning below is an experiment design choice, not a calibrated
vehicle/sensor specification. ``clearance`` must include any caller-required
estimation-error allowance; a sensor standard deviation is not a hard bound.
"""
from collections import OrderedDict
import hashlib
import heapq

import numpy as np

DOMAIN_SIDE = 800.
BIN_GRID = 8
BIN_SIDE = DOMAIN_SIDE/BIN_GRID
NAV_STANDOFF_M = 1.
POLYGON_PAD_M = .25
POLYGON_VERTICES = 24
WAYPOINT_REACHED_M = .75
TIE_BAND_M = .5
LOCAL_LOOKAHEAD_S = 6.
DEFAULT_OBSTACLES = ((220., 220., 12.), (300., 250., 15.), (450., 300., 15.),
                     (300., 450., 15.), (500., 550., 15.))
_CACHE = OrderedDict()


def _obstacles(obstacles):
    circles = np.asarray(obstacles, dtype=float)
    if circles.size == 0:
        return np.empty((0, 3), dtype=float)
    if circles.ndim != 2 or circles.shape[1] != 3 or not np.isfinite(circles).all() or np.any(circles[:, 2] <= 0):
        raise ValueError('obstacles must be finite [center_x, center_y, positive_radius] rows')
    return circles


def _point_free(points, circles, inflated, margin):
    points = np.asarray(points, dtype=float)
    valid = np.all((points >= margin) & (points <= DOMAIN_SIDE-margin), axis=-1)
    for center, radius in zip(circles[:, :2], inflated):
        valid &= np.sum((points-center)**2, axis=-1) >= radius*radius
    return valid


def _visible(first, second, circles, inflated, margin):
    """Broadcasted exact straight-segment/circle screening in estimated space."""
    first, second = np.broadcast_arrays(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    valid = _point_free(first, circles, inflated, margin) & _point_free(second, circles, inflated, margin)
    delta = second-first
    denominator = np.sum(delta*delta, axis=-1)
    for center, radius in zip(circles[:, :2], inflated):
        fraction = np.clip(np.sum((center-first)*delta, axis=-1)/np.maximum(denominator, 1e-30), 0., 1.)
        closest = first+fraction[..., None]*delta
        valid &= np.sum((closest-center)**2, axis=-1) >= radius*radius
    return valid


def _anchor(goal, nodes, circles, inflated, margin):
    lower = np.array([goal % BIN_GRID, goal//BIN_GRID])*BIN_SIDE
    center = lower+BIN_SIDE/2
    axis = np.linspace(1., BIN_SIDE-1., 11)
    xx, yy = np.meshgrid(axis, axis)
    candidates = np.vstack((center[None], lower+np.column_stack((xx.ravel(), yy.ravel())), nodes))
    within = np.all((candidates >= lower) & (candidates < lower+BIN_SIDE), axis=1)
    candidates = candidates[within & _point_free(candidates, circles, inflated, margin)]
    if not len(candidates):
        return None
    order = np.lexsort((candidates[:, 1], candidates[:, 0], np.sum((candidates-center)**2, axis=1)))
    return candidates[order[0]].copy()


class _Geometry:
    def __init__(self, circles, radius, clearance):
        self.circles = circles.copy()
        self.margin = float(radius+clearance)
        self.physical_inflated = circles[:, 2]+self.margin
        self.inflated = self.physical_inflated+NAV_STANDOFF_M
        angle = np.arange(POLYGON_VERTICES)*2*np.pi/POLYGON_VERTICES
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        vertices = []
        for center, inflated in zip(circles[:, :2], self.inflated):
            circumradius = (inflated+POLYGON_PAD_M)/np.cos(np.pi/POLYGON_VERTICES)
            vertices.extend(center+circumradius*direction)
        nodes = np.asarray(vertices, dtype=float).reshape(-1, 2)
        self.nodes = nodes[_point_free(nodes, circles, self.inflated, self.margin)]
        if len(self.nodes):
            self.visible = _visible(self.nodes[:, None], self.nodes[None], circles, self.inflated, self.margin)
            self.weights = np.where(self.visible, np.linalg.norm(self.nodes[:, None]-self.nodes[None], axis=-1), np.inf)
            np.fill_diagonal(self.weights, np.inf)
        else:
            self.visible = np.empty((0, 0), dtype=bool)
            self.weights = np.empty((0, 0))
        self.routes = {}

    def goal_route(self, goal, local=False):
        key = (int(goal), bool(local))
        if key in self.routes:
            return self.routes[key], True
        anchor = _anchor(int(goal), self.nodes, self.circles, self.inflated, self.margin)
        if anchor is None:
            self.routes[key] = (None, np.full(len(self.nodes), np.inf), np.zeros(len(self.nodes), dtype=bool))
            return self.routes[key], False
        allowed_nodes = np.ones(len(self.nodes), dtype=bool)
        if local:
            lower = np.array([goal % BIN_GRID, goal//BIN_GRID])*BIN_SIDE
            allowed_nodes &= np.all((self.nodes >= lower) & (self.nodes < lower+BIN_SIDE), axis=1)
        direct = _visible(self.nodes, anchor, self.circles, self.inflated, self.margin) & allowed_nodes
        distance = np.where(direct, np.linalg.norm(self.nodes-anchor, axis=1), np.inf)
        queue = [(float(distance[index]), int(index)) for index in np.flatnonzero(np.isfinite(distance))]
        heapq.heapify(queue)
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            candidates = np.flatnonzero(np.isfinite(self.weights[node]) & allowed_nodes)
            for previous in candidates:
                value = cost+self.weights[node, previous]
                if value < distance[previous]:
                    distance[previous] = value
                    heapq.heappush(queue, (float(value), int(previous)))
        self.routes[key] = (anchor, distance, allowed_nodes)
        return self.routes[key], False


def clear_navigation_cache():
    """Only performance caches are stored; there is no physical/planner state."""
    _CACHE.clear()


def _geometry(circles, radius, clearance):
    if not np.isfinite([radius, clearance]).all() or radius <= 0 or clearance < 0:
        raise ValueError('Positive radius and nonnegative navigation clearance required')
    signature = hashlib.sha256(circles.tobytes()+np.asarray([radius, clearance], dtype=float).tobytes()).hexdigest()
    hit = signature in _CACHE
    if not hit:
        _CACHE[signature] = _Geometry(circles, radius, clearance)
        if len(_CACHE) > 4:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(signature)
    return _CACHE[signature], hit


def navigate_nominal(p_hat, goals, nominal, obstacles, radius, clearance):
    """Return detoured references using estimated positions and known circles.

    Unreached robots follow a shared circumscribed-polygon visibility graph.
    Reached bins without obstacles (in particular the unblocked target support)
    retain the incoming nominal exactly. An obstacle-bearing reached transit bin
    uses a SAME-bin graph/anchor only when its nominal look-ahead is obstructed.
    The 1 m/s reference bound matches this validation's declared motion model.
    """
    positions, reference = np.asarray(p_hat, dtype=float), np.asarray(nominal, dtype=float)
    bins_goal = np.asarray(goals)
    if (positions.ndim != 2 or positions.shape[1:] != (2,) or reference.shape != positions.shape
            or bins_goal.shape != (len(positions),) or not np.issubdtype(bins_goal.dtype, np.integer)
            or not np.isfinite([positions, reference]).all() or np.any((bins_goal < 0) | (bins_goal >= 64))):
        raise ValueError('Finite matching N-by-2 positions/references and N integer goal-bin IDs required')
    speed = np.linalg.norm(reference, axis=1)
    if len(speed) and speed.max() > 1.+1e-8:
        raise ValueError('This validation navigator expects nominal velocity references capped at 1 m/s')
    circles = _obstacles(obstacles)
    command = reference.copy()
    info = dict(variant='Estimated-state visibility detour with explicit same-bin transit repair',
                obstacle_count=len(circles), changed_commands=0, detour_commands=0, reached_transit_repairs=0,
                modified_count=0, modified_robots=0,
                inflated_recovery_commands=0, blocked_ids=[], geometry_cache_hit=False, goal_cache_hits=0,
                navigation_standoff_m=NAV_STANDOFF_M, polygon_pad_m=POLYGON_PAD_M,
                polygon_vertices=POLYGON_VERTICES, tie_band_m=TIE_BAND_M,
                reached_repair_enabled=True, waypoint_samples=[])
    if not len(circles) or not len(positions):
        return command, info
    geometry, info['geometry_cache_hit'] = _geometry(circles, radius, clearance)
    current = np.floor(positions/BIN_SIDE).astype(int)
    current = current[:, 0]+BIN_GRID*current[:, 1]
    reached = current == bins_goal
    for goal in np.unique(bins_goal):
        all_ids = np.flatnonzero(bins_goal == goal)
        lower = np.array([goal % BIN_GRID, goal//BIN_GRID])*BIN_SIDE
        to_box = np.maximum(np.maximum(lower-circles[:, :2], circles[:, :2]-(lower+BIN_SIDE)), 0.)
        obstacle_in_goal = np.any(np.linalg.norm(to_box, axis=1) <= geometry.inflated+POLYGON_PAD_M)
        for local in [False, True]:
            indices = all_ids[reached[all_ids] == local]
            if local:
                if not obstacle_in_goal:
                    continue  # Positive target bins in the approved layout stay unchanged.
                free_lookahead = _visible(positions[indices], positions[indices]+LOCAL_LOOKAHEAD_S*reference[indices],
                                          circles, geometry.inflated, geometry.margin)
                indices = indices[~free_lookahead]
                info['reached_transit_repairs'] += len(indices)
            if not len(indices):
                continue
            (anchor, cost_to_goal, allowed_nodes), hit = geometry.goal_route(int(goal), local=local)
            info['goal_cache_hits'] += int(hit)
            if anchor is None:
                command[indices] = 0.
                info['blocked_ids'].extend(indices.tolist())
                continue
            points = positions[indices]
            free_points = _point_free(points, circles, geometry.inflated, geometry.margin)
            direct = free_points & _visible(points, anchor, circles, geometry.inflated, geometry.margin)
            original_clear = _visible(points, points+LOCAL_LOOKAHEAD_S*reference[indices],
                                       circles, geometry.inflated, geometry.margin)
            # Leave APF/repulsion references untouched when neither the goal
            # route nor their own near-term direction encounters an obstacle.
            direct_indices = indices[direct & (local | ~original_clear)]
            if len(direct_indices):
                difference = anchor-positions[direct_indices]
                distance = np.linalg.norm(difference, axis=1)
                desired = np.minimum(speed[direct_indices], 1.)
                if local:
                    desired = np.maximum(desired, .25)
                command[direct_indices] = difference/np.maximum(distance[:, None], 1e-30)*np.minimum(desired, distance/.5)[:, None]
            detour_indices = indices[free_points & ~direct]
            if len(detour_indices):
                distance = np.linalg.norm(geometry.nodes[None]-positions[detour_indices, None], axis=-1)
                visible = _visible(positions[detour_indices, None], geometry.nodes[None],
                                   circles, geometry.inflated, geometry.margin) & allowed_nodes[None]
                feasible = visible & np.isfinite(cost_to_goal)[None] & (distance > WAYPOINT_REACHED_M)
                total = np.where(feasible, distance+cost_to_goal[None], np.inf)
                minimum = total.min(axis=1) if total.shape[1] else np.full(len(detour_indices), np.inf)
                found = np.isfinite(minimum)
                blocked = detour_indices[~found]
                command[blocked] = 0.
                info['blocked_ids'].extend(blocked.tolist())
                if found.any():
                    tied = feasible[found] & (total[found] <= minimum[found, None]+TIE_BAND_M)
                    selected = np.argmax(tied, axis=1)
                    identities = detour_indices[found]
                    waypoints = geometry.nodes[selected]
                    difference = waypoints-positions[identities]
                    remaining = np.linalg.norm(difference, axis=1)
                    desired = np.minimum(speed[identities], 1.)
                    if local:
                        desired = np.maximum(desired, .25)
                    command[identities] = difference/np.maximum(remaining[:, None], 1e-30)*np.minimum(desired, remaining/.5)[:, None]
                    info['detour_commands'] += len(identities)
                    for identity, waypoint in zip(identities[:8], waypoints[:8]):
                        if len(info['waypoint_samples']) < 8:
                            info['waypoint_samples'].append(dict(id=int(identity), goal_bin=int(goal), local=local,
                                                                waypoint=waypoint.tolist()))
            for identity in indices[~free_points]:
                point = positions[identity]
                base_speed = min(float(speed[identity]), 1.)
                # An estimated point can enter an inflated uncertainty band.
                # Move its REFERENCE outward; no state is projected/reset.
                direction = np.zeros(2)
                for center, inflated in zip(circles[:, :2], geometry.inflated):
                    offset = point-center; length = float(np.linalg.norm(offset))
                    if length < inflated:
                        if length < 1e-12:
                            offset = anchor-center
                            if np.linalg.norm(offset) < 1e-12:
                                offset = np.array([1., 0.])
                        direction += offset/max(float(np.linalg.norm(offset)), 1e-30)*(inflated-length+1.)
                direction += np.maximum(geometry.margin-point, 0.)-np.maximum(point-(DOMAIN_SIDE-geometry.margin), 0.)
                if local:
                    destination = np.clip(point+direction, lower+1e-6, lower+BIN_SIDE-1e-6)
                    direction = destination-point
                length = float(np.linalg.norm(direction))
                command[identity] = direction/max(length, 1e-30)*max(base_speed, .25) if length else 0.
                info['inflated_recovery_commands'] += 1
    info['changed_commands'] = int(np.count_nonzero(np.linalg.norm(command-reference, axis=1) > 1e-12))
    info['modified_count'] = info['modified_robots'] = info['changed_commands']
    info['blocked_ids'] = sorted(set(info['blocked_ids']))
    return command, info


def admissible_matrix(matrix, p_hat, phi, obstacles):
    """Preserve original bin probabilities: obstacles are bypassed geometrically.

    A blocked direct center-to-center segment is not a blocked bin adjacency.
    No source/receiving mask or stochastic normalization is silently changed.
    Layout connectivity must pass ``validate_obstacle_layout`` before a run.
    """
    value = np.asarray(matrix)
    if value.shape != (64, 64) or not np.isfinite(value).all() or np.asarray(phi).shape != (64,):
        raise ValueError('A finite 64-by-64 transition matrix and 64-bin target are required')
    _obstacles(obstacles)
    if np.asarray(p_hat).ndim != 2 or np.asarray(p_hat).shape[1] != 2:
        raise ValueError('N-by-2 estimated positions required')
    return value.copy()


def validate_obstacle_layout(phi, initial_positions, obstacles=DEFAULT_OBSTACLES, radius=.5, clearance=.05):
    """Read-only preflight: initial clearance, target support and global routes."""
    target, initial, circles = np.asarray(phi, dtype=float), np.asarray(initial_positions, dtype=float), _obstacles(obstacles)
    if target.shape != (64,) or initial.ndim != 2 or initial.shape[1] != 2:
        raise ValueError('64-bin target and N-by-2 initial positions required')
    geometry, _ = _geometry(circles, radius, clearance)
    physical_clearance = float(np.min(np.linalg.norm(initial[:, None]-circles[None, :, :2], axis=2)
                                      -circles[None, :, 2]-radius)) if len(circles) else float('inf')
    positive = np.flatnonzero(target > 0)
    lower = np.column_stack((positive % 8, positive//8))*BIN_SIDE
    distance = np.linalg.norm(np.maximum(np.maximum(lower[:, None]-circles[None, :, :2],
                               circles[None, :, :2]-(lower[:, None]+BIN_SIDE)), 0.), axis=2)
    target_gap = float(np.min(distance-(geometry.inflated+POLYGON_PAD_M)[None])) if len(circles) and len(positive) else float('inf')
    centers = np.column_stack((np.arange(64) % 8, np.arange(64)//8))*BIN_SIDE+BIN_SIDE/2
    connected = True
    missing = []
    for goal in range(64):
        (anchor, costs, allowed), _ = geometry.goal_route(goal)
        if anchor is None:
            connected = False; missing.append(goal); continue
        direct = _visible(centers, anchor, circles, geometry.inflated, geometry.margin)
        if len(geometry.nodes):
            via = _visible(centers[:, None], geometry.nodes[None], circles, geometry.inflated, geometry.margin)
            direct |= np.any(via & np.isfinite(costs)[None], axis=1)
        if not direct.all():
            connected = False; missing.append(goal)
    return dict(layout_valid=bool(physical_clearance >= clearance and target_gap > 0 and connected),
                physical_initial_obstacle_clearance_m=physical_clearance,
                navigation_inflated_target_support_gap_m=target_gap,
                all_64_bin_center_routes_connected=bool(connected), blocked_goal_bins=missing,
                actual_initial_bins=np.unique(np.floor(initial[:, 0]/100).astype(int)+8*np.floor(initial[:, 1]/100).astype(int)).tolist(),
                obstacle_count=len(circles), obstacle_layout=circles.tolist(),
                notes='Navigation tuning and obstacle layout are controlled experiment assumptions, not empirical hardware calibration. Safety is checked separately on actual executed arcs.')

