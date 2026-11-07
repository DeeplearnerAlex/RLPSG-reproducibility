"""Reduced horizontal velocity response, NOT a calibrated UAV flight model.

The planner supplies a held velocity command. Only this execution model changes;
no post-integration position projection, teleportation, or collision correction.
Acceleration is held constant over each integration substep. Pair clearances are
minimized along the resulting parabolas, not just checked at their endpoints.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def norm_clip(vector, limit):
    vector = np.asarray(vector, dtype=float)
    return vector * np.minimum(1., limit / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-30))


@dataclass(frozen=True)
class VelocityResponse:
    tau_s: float = .5
    speed_cap_m_s: float = 1.
    acceleration_cap_m_s2: float = 1.
    integration_dt_s: float = .01

    def __post_init__(self):
        values = [self.tau_s, self.speed_cap_m_s, self.acceleration_cap_m_s2, self.integration_dt_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('All velocity response parameters must be finite and positive')
        if self.integration_dt_s > self.tau_s / 5:
            raise ValueError('Integration dt must be no larger than tau/5')

    def advance(self, positions, velocities, command, duration_s):
        """Return position, velocity, and (p0,v0,a,dt) substeps for auditing.

        a = clip_norm((clip_norm(command,vmax)-v)/tau, amax).
        A convex-ball speed projection changes acceleration, not position. The
        projected acceleration still respects amax when the initial v is valid.
        """
        p, v, command = map(lambda x: np.array(x, dtype=float, copy=True), (positions, velocities, command))
        if p.shape != v.shape or command.shape != p.shape or p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Expected equal N-by-2 arrays')
        if not len(p) or not np.isfinite(duration_s) or not np.isfinite([p, v, command]).all() or duration_s <= 0:
            raise ValueError('Finite states and positive duration required')
        if np.linalg.norm(v, axis=1).max() > self.speed_cap_m_s + 1e-9:
            raise ValueError('Initial velocity exceeds speed cap')
        command = norm_clip(command, self.speed_cap_m_s)
        steps = int(np.ceil(duration_s / self.integration_dt_s))
        h = duration_s / steps
        arcs = []
        for _ in range(steps):
            acceleration = norm_clip((command-v) / self.tau_s, self.acceleration_cap_m_s2)
            next_v = norm_clip(v + h*acceleration, self.speed_cap_m_s)
            acceleration = (next_v-v) / h
            arcs.append((p.copy(), v.copy(), acceleration.copy(), h))
            p = p + h*v + .5*h*h*acceleration
            v = next_v
        return p, v, arcs


def accelerated_pair_metrics(p, v, acceleration, dt, contact_distance):
    """Exact pair minimum for the piecewise-constant-acceleration model.

    Broad phase uses conservative displacement bounds. The minimum squared
    separation occurs at an endpoint or a real root of its cubic derivative.
    """
    if len(p) < 2:
        return float('inf'), np.empty((0, 2), dtype=int)
    tree = cKDTree(p)
    initial_min = float(tree.query(p, k=2)[0][:, 1].min())
    bound = float((np.linalg.norm(v, axis=1)*dt + .5*np.linalg.norm(acceleration, axis=1)*dt**2).max())
    pairs = tree.query_pairs(max(initial_min, contact_distance) + 2*bound + 1e-9, output_type='ndarray')
    i, j = pairs.T
    r, u, b = p[i]-p[j], v[i]-v[j], .5*(acceleration[i]-acceleration[j])
    ends = r+dt*u+dt**2*b
    squared = np.minimum(np.sum(r*r, axis=1), np.sum(ends*ends, axis=1))
    c0 = np.sum(r*u, axis=1)
    c1 = np.sum(u*u+2*r*b, axis=1)
    c2 = 3*np.sum(u*b, axis=1)
    c3 = 2*np.sum(b*b, axis=1)
    # If the derivative cannot change sign, endpoint checks are exact. This
    # vectorized certificate avoids thousands of redundant cubic solves.
    variation = abs(c1)*dt+abs(c2)*dt**2+abs(c3)*dt**3
    possible_roots = (abs(c0) <= variation+1e-20) & ((abs(c1)+abs(c2)+abs(c3)) > 0)
    for k in np.flatnonzero(possible_roots):
        coefficients = np.trim_zeros(np.array([c3[k], c2[k], c1[k], c0[k]]), 'f')
        candidates = []
        if len(coefficients) > 1:
            for root in np.roots(coefficients):
                if abs(root.imag) < 1e-9 and 0 < root.real < dt:
                    candidates.append(float(root.real))
        if candidates:
            t = np.asarray(candidates)
            curve = r[k]+t[:, None]*u[k]+t[:, None]**2*b[k]
            squared[k] = min(squared[k], float(np.sum(curve*curve, axis=1).min()))
    distances = np.sqrt(squared)
    return float(min(initial_min, distances.min())), pairs[distances < contact_distance-1e-8]


def finite_population_hd_floor(phi, count):
    """Closest attainable count vector for Hellinger error (greedy concave gain)."""
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or not len(phi) or not np.isfinite(phi).all() or (phi < 0).any() or phi.sum() <= 0:
        raise ValueError('phi must be a finite nonnegative probability vector with positive mass')
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    phi = phi / phi.sum()
    counts = np.zeros(len(phi), dtype=int)
    for _ in range(count):
        gain = np.sqrt(phi) * (np.sqrt(counts+1)-np.sqrt(counts))
        counts[np.argmax(gain)] += 1
    hd = float(np.linalg.norm(np.sqrt(counts/count)-np.sqrt(phi))/np.sqrt(2))
    return hd, counts


def accelerated_boundary_margin(p, v, acceleration, dt, radius, domain_side):
    """Minimum distance-to-wall minus radius on each parabolic subarc."""
    end = p+dt*v+.5*dt**2*acceleration
    lower = np.minimum(p, end)
    upper = np.maximum(p, end)
    turning = np.divide(-v, acceleration, out=np.zeros_like(v), where=abs(acceleration)>1e-30)
    active = (abs(acceleration)>1e-30) & (turning > 0) & (turning < dt)
    stationary = p+turning*v+.5*turning**2*acceleration
    lower = np.where(active, np.minimum(lower, stationary), lower)
    upper = np.where(active, np.maximum(upper, stationary), upper)
    return float(min(lower.min()-radius, domain_side-radius-upper.max()))

