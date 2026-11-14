"""Safe Passage sampling planner: a randomised second planner beside the lattice of safe_route.py.

plan_route() has the lattice's signature and returns the same RoutePlan (statuses, waypoints, circles, goals); the
extra arguments are keyword-only (seed, time_budget, sampler).

How it works
  A forest of straight legs grown at random (RRT with a choice of parent), one search per margin level of
  RouteParams.levels, strictest first, the time budget shared between the levels.
  node      a point, the heading the boat arrives with, and which credits its path has earned. The credits are kept
            per heading offset (0, +tol, -tol: the fine check wants every one of them), six bits; the "layer" of a
            node is (red at all offsets, green at all offsets). A sample may add a node in every layer.
  edge      the turn on the spot at the parent, then the straight leg. Checked by _Fast, an analytic, batched version
            of the strict rule for exactly that motion (the checker of safe_route walks poses one at a time in Python:
            30 ms per route, too slow for tens of thousands of edges). _Fast only steers the search: every route that
            is handed out has passed safe_route._acceptable (PassTracker at the three heading offsets, clearance,
            doubling back), before and after simplification.
  roots     points on the entry circle(s) with the heading of the loop's last chord, or the given start.
  goal      any point of an exit circle reached within max_turn of its counterclockwise direction, as the lattice.
  samples   uniform over the lattice's water, plus points near the red/green buoys that can still give a credit,
            between red/green pairs, and on the exit circle.
  cost      the lattice's: length x (1 + soft clearance cost) + corner_cost + turn_cost x turn.
  after     the cheapest routes that reach the exit are fine-checked, simplified (greedy corner shortcuts as the
            lattice does, or validated Ramer-Douglas-Peucker), assembled like _plan_level does, and one is chosen
            between the levels with the lattice's Monte Carlo rule.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:                                        # inside the package
    from . import safe_route as sr
except ImportError:                         # stand-alone (frames on sys.path)
    from frames.utils import safe_route as sr

from_sr = sr                                # noqa: the shared pieces all come from safe_route
Buoy, RouteParams, RoutePlan = sr.Buoy, sr.RouteParams, sr.RoutePlan
ENTRY, EXIT, RED, GREEN = sr.ENTRY, sr.EXIT, sr.RED, sr.GREEN
wrap = sr.wrap


@dataclass
class SamplerParams:
    parents: str = "best"            # "best": cheapest valid parent among the nearest of every layer (RRT* without
    #                                  rewiring); "nearest": the one nearest node (plain RRT)
    eta: float = 6.0                 # m, longest leg grown in one step
    near_factor: float = 1.3         # parents are looked for within near_factor x eta of the new point
    k_layer: int = 3                 # parents tried per layer
    p_goal: float = 0.08             # sample on the exit circle
    p_credit: float = 0.25           # sample near a buoy that can give a credit still needed
    p_gate: float = 0.12             # sample between a red and a green
    goal_points: int = 16            # points of each exit circle offered to a new node
    goal_tries: int = 5              # of which the nearest this many are checked
    roots: int = 24                  # start points per entry circle
    step: float = 0.3                # m, sampling of a leg in the fast check
    batch: int = 16                  # samples per batch (one call of the fast check)
    levels: Optional[tuple] = None   # margin levels searched, in this order; None: all, strictest first
    all_levels: bool = False         # False: stop at the first level whose route passes the Monte Carlo; True: search
    #                                  every level and choose between them as the lattice does
    simplify: str = "polish+shortcut+rdp"   # steps joined by "+": "polish" (random shortcuts between points of the
    #                                  path), "shortcut" (the lattice's greedy corner shortcuts), "rdp", "none"
    polish_rounds: int = 3
    polish_trials: int = 60
    rdp_eps: tuple = (3.0, 2.0, 1.0, 0.5, 0.25)   # m, tolerances tried, largest first; first acceptable one is kept
    max_mc: int = 4                  # Monte Carlo runs at most (each costs about a second)
    max_validate: int = 8            # routes fine-checked per level and ask, cheapest first
    improve: float = 1.0             # after the first validated route of a level keep searching this share of the
    #                                  time it took (at least improve_min s), then stop that level
    improve_min: float = 0.3
    max_nodes: int = 40000
    max_iterations: Optional[int] = None   # per level; with a seed and this, a run is reproducible


# ----------------------------------------------------------------------------------------------------------------
# Ramer-Douglas-Peucker, and validated simplification
# ----------------------------------------------------------------------------------------------------------------

def rdp(points, eps: float) -> list:
    """Indices of the points Ramer-Douglas-Peucker keeps at tolerance eps."""
    P = np.asarray([p[:2] for p in points], dtype=float)
    keep = np.zeros(len(P), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = float(ab @ ab)
        rel = P[i + 1:j] - a
        if L2 < 1e-12:
            d = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip(rel @ ab / L2, 0.0, 1.0)
            d = np.linalg.norm(rel - t[:, None] * ab[None], axis=1)
        k = int(d.argmax())
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return [int(i) for i in np.nonzero(keep)[0]]


def simplify_rdp(pts, acceptable, eps_list=(3.0, 2.0, 1.0, 0.5, 0.25), got=None):
    """RDP at the largest tolerance whose result still passes the fine check. Returns (pts, got, eps used or None).
    A simplified path is only ever returned after acceptable() has passed it."""
    for eps in eps_list:
        idx = rdp(pts, eps)
        if len(idx) == len(pts):
            continue
        trial = [pts[i] for i in idx]
        g = acceptable(trial)
        if g is not None:
            return trial, g, eps
    return pts, got, None


def simplify_shortcut(pts, acceptable, got=None, prefilter=None):
    """The lattice's corner shortcuts: from each corner, the farthest later corner that can be driven to directly
    with the fine check still holding. prefilter(trial) -> False skips a trial that cannot pass (speed only)."""
    i = 0
    while i < len(pts) - 2:
        j = len(pts) - 1
        while j > i + 1:
            trial = pts[:i + 1] + pts[j:]
            if prefilter is None or prefilter(trial):
                g = acceptable(trial)
                if g is not None:
                    pts, got = trial, g
                    break
            j -= 1
        i += 1
    return pts, got


# ----------------------------------------------------------------------------------------------------------------
# The strict rule for "turn on the spot, then drive a straight leg", batched
# ----------------------------------------------------------------------------------------------------------------

class _Fast:
    """Strict rule (one margin level) for edges: at A, turn from yaw_in to face B (the shorter way), drive to B.
    The same events PassTracker registers on densify()'s poses, found in closed form:
      on the leg   buoy b is abeam where along_b = 0: at t = (b - A).h / cos(off), h the heading (travel + off);
      on the turn  when the heading passes bearing_b -+ 90 deg;
      each event is classified as PassTracker.classify does (shield, rival, margins);
      "near" (a wrong-side buoy within near_range inside the beam sector) is a condition on poses: sampled on the
      leg, an interval test on the turn.
    for the three heading offsets of the fine check. Differences from the tracker (its 1 deg hysteresis, the 0.1 m
    sampling) are small and are caught by the fine check a route gets before it is handed out."""

    def __init__(self, q, p: RouteParams, level, step: float = 0.3):
        self.q, self.p, self.step = q, p, step
        self.clearance, self.shield_margin, self.credit_margin, self.near_range = level
        self.P = q.P
        self.B = len(q.P)
        self.red, self.green, self.trusted, self.confirmed = q.red, q.green, q.trusted, q.confirmed
        self.clear_b = q.clearances(level[0], p)
        tol = math.radians(p.heading_tolerance_deg)
        self.offs = np.array([0.0, tol, -tol])
        self.cos_o = np.cos(self.offs)
        self.cos_near = math.cos(math.radians(p.near_sector_deg))
        self.w_near = math.asin(min(1.0, self.cos_near))          # half-width of the beam sector the tracker uses
        self.notself = ~np.eye(self.B, dtype=bool)
        self.coloured = self.red | self.green
        self.max_turn = math.radians(p.max_turn_deg)
        self.calls = 0
        self.checked = 0

    def free(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, 2)
        ok = np.all(np.linalg.norm(self.P[None] - X[:, None], axis=2) >= self.clear_b[None], axis=1)
        if self.q.tree is not None:
            ok &= self.q.tree.query(X)[0] >= self.p.obstacle_clearance
        if self.q.keep_out is not None:
            ok &= ~np.asarray(self.q.keep_out(X), dtype=bool)
        return ok

    def _classify(self, X, Nrm, b):
        """Events (M of them): buoy b[m] abeam with the boat at X[m], port normal Nrm[m]. -> violation, credit (M,)"""
        p = self.p
        m = np.arange(len(b))
        d = self.P[None] - X[:, None, :]                      # (M, B, 2)
        dist = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
        lat = d[..., 0] * Nrm[:, None, 0] + d[..., 1] * Nrm[:, None, 1]
        db = dist[m, b]
        s = np.where(lat[m, b] > 0, 1.0, -1.0)
        ls = lat * s[:, None]
        other = self.notself[b]
        shield = np.where(other & self.trusted & (ls >= p.shield_lat_min), dist, np.inf).min(axis=1)
        rival = np.where(other & (ls > -p.credit_lat_slack), dist, np.inf).min(axis=1)
        port = s > 0
        red, green = self.red[b], self.green[b]
        wrong = (port & red) | (~port & green)
        right = (~port & red) | (port & green)
        shielded = (db - shield >= self.shield_margin) & (db > self.near_range)
        viol = wrong & (db <= p.violation_range) & ~shielded
        credit = right & self.confirmed[b] & (db <= p.credit_range) & (rival - db >= self.credit_margin)
        return viol, credit

    def edges(self, A, Bp, yaw_in):
        """A, Bp (E, 2); yaw_in (E,) heading on arrival at A (nan: already facing the leg, nothing judged in the
        turn). -> valid (E,), bits (E,) int (red credit at offset o: bit o; green: bit 3 + o), cost (E,),
        turn (E,) signed, length (E,)."""
        p = self.p
        A = np.asarray(A, dtype=float).reshape(-1, 2)
        Bp = np.asarray(Bp, dtype=float).reshape(-1, 2)
        yaw_in = np.asarray(yaw_in, dtype=float).reshape(-1)
        E = len(A)
        self.calls += 1
        self.checked += E
        dv = Bp - A
        L = np.hypot(dv[:, 0], dv[:, 1])
        Ls = np.maximum(L, 1e-9)
        u = dv / Ls[:, None]
        th = np.arctan2(dv[:, 1], dv[:, 0])
        has_turn = ~np.isnan(yaw_in)
        yi = np.where(has_turn, yaw_in, th)
        delta = wrap(th - yi)
        absd = np.abs(delta)
        valid = (L > 1e-3) & (absd <= self.max_turn + 1e-9)

        # clearance: exact distance from every buoy to the leg
        rel = self.P[None] - A[:, None]                                        # (E, B, 2)
        tb = np.clip(rel[..., 0] * u[:, None, 0] + rel[..., 1] * u[:, None, 1], 0.0, L[:, None])
        dseg = np.hypot(rel[..., 0] - tb * u[:, None, 0], rel[..., 1] - tb * u[:, None, 1])
        valid &= np.all(dseg >= self.clear_b - 1e-9, axis=1)
        # the leg, sampled: obstacles, soft cost, near rule
        n = max(2, int(math.ceil(float(L.max()) / self.step)) + 1)
        f = np.linspace(0.0, 1.0, n)
        S = A[:, None, :] + dv[:, None, :] * f[None, :, None]                 # (E, n, 2)
        D = self.P[None, None] - S[:, :, None, :]                              # (E, n, B, 2)
        dist = np.sqrt(D[..., 0] ** 2 + D[..., 1] ** 2)                        # (E, n, B)
        if self.q.tree is not None:
            lim = p.obstacle_clearance + 0.03
            valid &= np.isinf(self.q.tree.query(S.reshape(-1, 2), distance_upper_bound=lim)[0]).reshape(E, n).all(axis=1)
        if self.q.keep_out is not None:
            valid &= ~np.asarray(self.q.keep_out(S.reshape(-1, 2)), dtype=bool).reshape(E, n).any(axis=1)
        dmin = dist.min(axis=2) if self.B else np.full((E, n), np.inf)
        soft = p.soft_weight * (np.clip(p.soft_clearance - dmin, 0.0, None)
                                / max(p.soft_clearance - self.clearance, 1e-6)) ** 2
        cost = L * (1.0 + soft.mean(axis=1)) + np.where(has_turn & (absd > 1e-6),
                                                        p.turn_cost * absd + p.corner_cost, 0.0)
        psi = th[:, None] + self.offs[None]                                    # (E, 3)
        hx, hy = np.cos(psi), np.sin(psi)
        if self.near_range > 0 and self.B:
            ee, kk, bb = np.nonzero((dist <= self.near_range) & self.coloured)  # the few close poses: all 3 offsets
            if len(ee):
                dx, dy, dd = D[ee, kk, bb, 0], D[ee, kk, bb, 1], dist[ee, kk, bb]
                along = dx[:, None] * hx[ee] + dy[:, None] * hy[ee]             # (M, 3)
                lat = -dx[:, None] * hy[ee] + dy[:, None] * hx[ee]
                wrong = ((lat > 0) & self.red[bb, None]) | ((lat < 0) & self.green[bb, None])
                bad = (wrong & (np.abs(along) <= self.cos_near * dd[:, None])).any(axis=1)
                valid[ee[bad]] = False

        # abeam events on the leg and in the turn at A, gathered: (edge, offset, buoy)
        relh = rel[:, None, :, 0] * hx[:, :, None] + rel[:, None, :, 1] * hy[:, :, None]     # (E, 3, B)
        tstar = relh / self.cos_o[None, :, None]
        act_leg = (tstar >= 0.0) & (tstar <= L[:, None, None]) & valid[:, None, None]
        e1, o1, b1 = np.nonzero(act_leg)
        X1 = A[e1] + tstar[e1, o1, b1][:, None] * u[e1]
        N1 = np.stack([-hy[e1, o1], hx[e1, o1]], axis=1)
        dA = np.sqrt(rel[..., 0] ** 2 + rel[..., 1] ** 2)                      # (E, B)
        beta = np.arctan2(rel[..., 1], rel[..., 0])
        sgn = np.where(delta >= 0, 1.0, -1.0)
        y0 = yi[:, None, None] + self.offs[None, :, None]                      # (E, 3, 1) heading as the turn begins
        xp = wrap(beta[:, None, :] - math.pi / 2 - y0) * sgn[:, None, None]    # port crossing at bearing - 90
        xs = wrap(beta[:, None, :] + math.pi / 2 - y0) * sgn[:, None, None]
        ht = (has_turn & valid)[:, None, None]
        ap = ht & (xp > 0) & (xp <= absd[:, None, None])
        as_ = ht & (xs > 0) & (xs <= absd[:, None, None])
        e2, o2, b2 = np.nonzero(ap | as_)
        side = np.where(ap[e2, o2, b2], 1.0, -1.0)
        N2 = side[:, None] * rel[e2, b2] / np.maximum(dA[e2, b2], 1e-9)[:, None]
        ev_e, ev_o, ev_b = np.concatenate([e1, e2]), np.concatenate([o1, o2]), np.concatenate([b1, b2])
        r = np.zeros((E, 3), dtype=bool)
        g = np.zeros((E, 3), dtype=bool)
        if len(ev_e):
            viol, credit = self._classify(np.concatenate([X1, A[e2]]), np.concatenate([N1, N2]), ev_b)
            valid[ev_e[viol]] = False
            cr, cg = credit & self.red[ev_b], credit & self.green[ev_b]
            r[ev_e[cr], ev_o[cr]] = True
            g[ev_e[cg], ev_o[cg]] = True
        # near rule in the turn: a red within near_range inside the port beam sector at some heading swept through
        if self.near_range > 0 and self.B:
            mid = y0 + (delta / 2.0)[:, None, None]
            half = self.w_near + (absd / 2.0)[:, None, None]
            close = (dA <= self.near_range)[:, None, :]
            badr = self.red & close & (np.abs(wrap(beta[:, None, :] - math.pi / 2 - mid)) <= half)
            badg = self.green & close & (np.abs(wrap(beta[:, None, :] + math.pi / 2 - mid)) <= half)
            valid &= ~(has_turn[:, None, None] & (badr | badg)).any(axis=(1, 2))
        w = np.array([1, 2, 4])
        bits = (r * w).sum(axis=1) + 8 * (g * w).sum(axis=1)
        return valid, bits.astype(np.int64), cost, delta, L


def _layer(bits):
    """(red at every offset) | 2 (green at every offset)."""
    bits = np.asarray(bits)
    return ((bits & 7) == 7).astype(np.int64) | (((bits >> 3) & 7) == 7).astype(np.int64) << 1


# ----------------------------------------------------------------------------------------------------------------
# One search at one margin level
# ----------------------------------------------------------------------------------------------------------------

class _Tree:
    def __init__(self, q, p: RouteParams, li: int, need_red: bool, need_green: bool, sp: SamplerParams, rng):
        self.q, self.p, self.li, self.sp, self.rng = q, p, li, sp, rng
        self.level = p.levels[li]
        self.fast = _Fast(q, p, self.level, sp.step)
        self.strict = replace(p, clearance=self.level[0], shield_margin=self.level[1], credit_margin=self.level[2],
                              near_range=self.level[3])
        self.clear_fine = q.clearances(self.level[0], p) - 0.1
        self.want = (1 if need_red else 0) | (2 if need_green else 0)
        self.bits0 = (0 if need_red else 7) | (0 if need_green else 56)     # a colour not looked for: as if in hand
        self.from_start = q.start is not None
        cap = 1024
        self.X = np.zeros((cap, 2))
        self.yaw = np.zeros(cap)            # heading on arrival (nan: a given start, facing any way)
        self.bits = np.zeros(cap, dtype=np.int64)
        self.layer = np.zeros(cap, dtype=np.int64)
        self.cost = np.zeros(cap)
        self.parent = np.full(cap, -1, dtype=np.int64)
        self.turn = np.zeros(cap)           # turn made at the parent before the leg to this node
        self.leg = np.zeros(cap)
        self.n = 0
        self.finishes = []                  # (cost, node, goal point or None, bits at the goal)
        self.iterations = 0
        self.new_full = False               # a finish with everything wanted has come in since this was cleared
        self.bad = set()                    # finishes that failed the fine check
        self.fine_calls = 0
        self.fragile = []                   # routes (corner points) that failed the Monte Carlo
        P = q.P
        self.lo = P.min(axis=0) - p.grid_margin
        self.hi = P.max(axis=0) + p.grid_margin
        if self.from_start:
            self.lo = np.minimum(self.lo, q.start - p.start_snap)
            self.hi = np.maximum(self.hi, q.start + p.start_snap)
        self.exit_xy = P[q.exit_] if q.exit_ is not None else None
        col = []
        if need_red:
            col += list(np.nonzero(q.red & q.confirmed)[0])
        if need_green:
            col += list(np.nonzero(q.green & q.confirmed)[0])
        self.credit_buoys = col
        reds, greens = np.nonzero(q.red)[0], np.nonzero(q.green)[0]
        self.gates = [(i, j) for i in reds for j in greens if np.linalg.norm(P[i] - P[j]) <= 14.0]
        self._roots()

    # -- nodes ----------------------------------------------------------------------------------------------------
    def _add(self, x, yaw, bits, cost, parent, turn, leg) -> int:
        if self.n == len(self.X):
            for name in ("X", "yaw", "bits", "layer", "cost", "parent", "turn", "leg"):
                a = getattr(self, name)
                b = np.zeros((2 * len(a),) + a.shape[1:], dtype=a.dtype)
                b[:len(a)] = a
                setattr(self, name, b)
        i = self.n
        self.X[i], self.yaw[i], self.bits[i], self.cost[i] = x, yaw, bits, cost
        self.parent[i], self.turn[i], self.leg[i] = parent, turn, leg
        self.layer[i] = int(_layer(bits))
        self.n += 1
        return i

    def _roots(self):
        q, p, sp = self.q, self.p, self.sp
        if self.from_start:
            pts = [q.start]
            for _ in range(40):
                a, r = self.rng.uniform(0, 2 * math.pi), p.start_snap * math.sqrt(self.rng.random())
                pts.append(q.start + r * np.array([math.cos(a), math.sin(a)]))
            pts = np.array(pts)
            ok = self.fast.free(pts)
            d0 = np.linalg.norm(pts - q.start[None], axis=1)
            if ok[0]:
                ok[1:] = False              # the start itself is allowed: the route begins there
            for i in np.nonzero(ok)[0][:12]:
                self._add(pts[i], math.nan, self.bits0, 1e-6 + 10.0 * d0[i], -1, 0.0, 0.0)
            return
        e = q.P[q.entry]
        for R in q.entry_radii:
            phi = self.rng.uniform(0, 2 * math.pi) + np.arange(sp.roots) * 2 * math.pi / sp.roots
            pts = e[None] + R * np.stack([np.cos(phi), np.sin(phi)], axis=1)
            ok = self.fast.free(pts)
            for i in np.nonzero(ok)[0]:
                self._add(pts[i], float(wrap(sr.arrival_yaw(phi[i], 1, p.circle_points))), self.bits0, 1e-6, -1,
                          0.0, 0.0)

    def _doubling(self, parent: int, turn: float) -> bool:
        """Would a turn of `turn` at node `parent` double back (the cumulative-turn rule of _doubling_back)?"""
        limit, window = math.radians(self.p.double_back_deg), self.p.double_back_window
        total, d, v = turn, 0.0, parent
        while self.parent[v] >= 0:
            d += self.leg[v]
            if d > window:
                break
            total += self.turn[v]
            if abs(total) > limit:
                return True
            v = self.parent[v]
        return False

    def path(self, node: int, goal=None) -> list:
        out = [] if goal is None else [np.array(goal, dtype=float)]
        v = node
        while v >= 0:
            out.append(self.X[v].copy())
            v = self.parent[v]
        return out[::-1]

    # -- sampling -------------------------------------------------------------------------------------------------
    def _sample(self):
        sp, rng, p, P = self.sp, self.rng, self.p, self.q.P
        r = rng.random()
        if r < sp.p_goal and self.exit_xy is not None:
            R, a = rng.choice(self.q.exit_radii), rng.uniform(0, 2 * math.pi)
            return self.exit_xy + (R + rng.uniform(0.0, 3.0)) * np.array([math.cos(a), math.sin(a)])
        r -= sp.p_goal
        if r < sp.p_credit and self.credit_buoys:
            b = self.credit_buoys[rng.integers(len(self.credit_buoys))]
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(self.level[0], p.credit_range + 1.0)
            return P[b] + d * np.array([math.cos(a), math.sin(a)])
        r -= sp.p_credit
        if r < sp.p_gate and self.gates:
            i, j = self.gates[rng.integers(len(self.gates))]
            return P[i] + rng.uniform(0.25, 0.75) * (P[j] - P[i]) + rng.normal(0.0, 1.0, 2)
        return self.lo + rng.random(2) * (self.hi - self.lo)

    # -- growth ---------------------------------------------------------------------------------------------------
    def step(self):
        """One batch: sp.batch samples, each steered from its nearest node, joined to the cheapest valid parent of
        every layer; all the edges of the batch go through the fast check in one call."""
        sp = self.sp
        self.iterations += 1
        if not self.n:
            return
        n0 = self.n
        X = self.X[:n0]
        S = np.array([self._sample() for _ in range(sp.batch)])
        Dm = np.hypot(S[:, None, 0] - X[None, :, 0], S[:, None, 1] - X[None, :, 1])
        i0 = Dm.argmin(axis=1)
        d0 = Dm[np.arange(len(S)), i0]
        far = d0 > sp.eta
        if far.any():
            S[far] = X[i0[far]] + (S[far] - X[i0[far]]) * (sp.eta / d0[far])[:, None]
            Dm[far] = np.hypot(S[far, None, 0] - X[None, :, 0], S[far, None, 1] - X[None, :, 1])
            d0 = np.minimum(d0, sp.eta)
        keep = np.nonzero((d0 >= 0.3) & self.fast.free(S))[0]
        cands, owner = [], []
        for s in keep:
            if sp.parents == "nearest":
                c = np.array([i0[s]])
            else:
                near = np.nonzero(Dm[s] <= sp.near_factor * sp.eta)[0]
                near = near[np.argsort(Dm[s, near])]
                lay = self.layer[near]
                c = np.concatenate([near[lay == k][:sp.k_layer] for k in range(4)])
            cands.append(c)
            owner.append(np.full(len(c), s))
        if not cands:
            return
        cand, owner = np.concatenate(cands), np.concatenate(owner)
        valid, bits, cost, turn, L = self.fast.edges(X[cand], S[owner], self.yaw[cand])
        if not valid.any():
            return
        bits = bits | self.bits[cand]
        total = self.cost[cand] + cost
        lay = _layer(bits)
        new = []
        for s in keep:
            mine = valid & (owner == s)
            if not mine.any():
                continue
            x = S[s]
            for k in range(4):
                idx = np.nonzero(mine & (lay == k))[0]
                for e in idx[np.argsort(total[idx])]:
                    c = int(cand[e])
                    if self._doubling(c, float(turn[e])):
                        continue
                    th = math.atan2(x[1] - X[c, 1], x[0] - X[c, 0])
                    new.append(self._add(x, th, int(bits[e]), float(total[e]), c, float(turn[e]), float(L[e])))
                    break
        if new:
            self._finish(np.array(new))

    def _finish(self, V):
        """Try to end the route from the nodes V: on an exit circle, or (no exit buoy) right there."""
        sp, p, q = self.sp, self.p, self.q
        if self.exit_xy is None:
            for v in V:
                if self.layer[v] & self.want == self.want and self.parent[v] >= 0:
                    self._record(float(self.cost[v]), int(v), None, int(self.bits[v]))
            return
        G, PHI = [], []
        for R in q.exit_radii:
            phi = self.rng.uniform(0, 2 * math.pi) + np.arange(sp.goal_points) * 2 * math.pi / sp.goal_points
            G.append(self.exit_xy[None] + R * np.stack([np.cos(phi), np.sin(phi)], axis=1))
            PHI.append(phi)
        G, PHI = np.concatenate(G), np.concatenate(PHI)
        XV = self.X[V]
        dv = G[None] - XV[:, None]                                           # (V, G, 2)
        L = np.hypot(dv[..., 0], dv[..., 1])
        off = np.abs(wrap(np.arctan2(dv[..., 1], dv[..., 0]) - (PHI[None] + math.pi / 2)))
        score = np.where((off <= math.radians(p.max_turn_deg) + 0.25) & (L > 0.2), L + p.turn_cost * off, np.inf)
        order = np.argsort(score, axis=1)[:, :sp.goal_tries]
        vi = np.repeat(np.arange(len(V)), order.shape[1])
        gi = order.ravel()
        ok = np.isfinite(score[vi, gi])
        vi, gi = vi[ok], gi[ok]
        if not len(vi):
            return
        valid, bits, cost, turn, _ = self.fast.edges(XV[vi], G[gi], self.yaw[V[vi]])
        for e in np.nonzero(valid)[0]:
            v = int(V[vi[e]])
            if self._doubling(v, float(turn[e])):
                continue
            self._record(float(self.cost[v] + cost[e] + p.turn_cost * off[vi[e], gi[e]]), v, G[gi[e]].copy(),
                         int(bits[e] | self.bits[v]))

    def _record(self, cost, v, goal, bits):
        self.finishes.append((cost, v, goal, bits))
        if int(_layer(bits)) & self.want == self.want:
            self.new_full = True

    # -- routes ---------------------------------------------------------------------------------------------------
    def start_yaw_of(self, points):
        first = points[0]
        if self.from_start:
            return math.atan2(points[1][1] - first[1], points[1][0] - first[0])
        e = self.q.P[self.q.entry]
        return sr.arrival_yaw(math.atan2(first[1] - e[1], first[0] - e[0]), 1, self.p.circle_points)

    def acceptable(self, points, need_red, need_green):
        if len(points) < 2:
            return None
        self.fine_calls += 1
        return sr._acceptable(points, self.start_yaw_of(points), self.q, self.clear_fine, self.p, self.strict,
                              need_red, need_green)

    def fast_route(self, points, need_red, need_green):
        """The fast check on a whole polyline -> its cost, or None when it fails (prefilter and polishing)."""
        pts = np.asarray(points, dtype=float)
        dv = np.diff(pts, axis=0)
        yaws = np.arctan2(dv[:, 1], dv[:, 0])
        yin = np.concatenate([[math.nan if self.from_start else self.start_yaw_of(points)], yaws[:-1]])
        valid, bits, cost, turn, L = self.fast.edges(pts[:-1], pts[1:], yin)
        if not valid.all():
            return None
        if int(_layer(np.bitwise_or.reduce(bits) | (0 if need_red else 7) | (0 if need_green else 56))) != 3:
            return None
        if sr._doubling_back(points, self.start_yaw_of(points), self.p):
            return None
        return float(cost.sum())

    def polish(self, pts, got, need_red, need_green):
        """Random shortcuts: two random points of the path joined by a straight leg when the fast check passes and
        the cost falls; after each round the fine check decides whether the round is kept."""
        sp, rng = self.sp, self.rng
        best_cost = self.fast_route(pts, need_red, need_green)
        if best_cost is None:           # the fast and the fine check disagree about this path: leave it alone
            return pts, got
        for _ in range(sp.polish_rounds):
            trial, cost, changed = list(pts), best_cost, False
            for _ in range(sp.polish_trials):
                P = np.asarray(trial)
                seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                a, b = np.sort(rng.uniform(0.0, cum[-1], 2))
                ia, ib = int(np.searchsorted(cum, a, side="right")) - 1, int(np.searchsorted(cum, b, side="right")) - 1
                ia, ib = min(ia, len(seg) - 1), min(ib, len(seg) - 1)
                if ia == ib:
                    continue
                pa = P[ia] + (P[ia + 1] - P[ia]) * ((a - cum[ia]) / max(seg[ia], 1e-9))
                pb = P[ib] + (P[ib + 1] - P[ib]) * ((b - cum[ib]) / max(seg[ib], 1e-9))
                head = trial[:ia + 1] + ([pa] if a - cum[ia] > 0.3 else [])
                tail = ([pb] if cum[ib + 1] - b > 0.3 else []) + trial[ib + 1:]
                new = head + tail
                if len(new) < 2:
                    continue
                c = self.fast_route(new, need_red, need_green)
                if c is not None and c < cost - 1e-6:
                    trial, cost, changed = new, c, True
            if not changed:
                break
            g = self.acceptable(trial, need_red, need_green)
            if g is not None:
                pts, got, best_cost = trial, g, cost
        return pts, got

    def best(self, need_red: bool, need_green: bool, stats=None, tries=None, simplify=None):
        """The cheapest finish that passes the fine check for this ask, simplified. -> (pts, (dense, res), raw pts)
        or None. Finishes that failed the fine check are remembered and not tried again."""
        sp = self.sp
        ask = ((1 if need_red else 0) | (2 if need_green else 0)) & self.want
        fin = sorted((f for f in self.finishes if int(_layer(f[3])) & ask == ask), key=lambda f: f[0])
        tried = 0

        def acc(points):
            return self.acceptable(points, need_red, need_green)

        for cost, v, goal, bits in fin:
            if tried >= (tries or sp.max_validate):
                break
            key = (need_red, need_green, v, None if goal is None else (round(goal[0], 3), round(goal[1], 3)))
            if key in self.bad:
                continue
            pts = self.path(v, goal)
            tried += 1
            got = acc(pts)
            if got is None and self.exit_xy is None and len(pts) > 1:
                # no exit buoy: the route stops as the last buoy comes abeam; run on so the pass is beyond doubt
                u = (pts[-1] - pts[-2]) / max(float(np.linalg.norm(pts[-1] - pts[-2])), 1e-9)
                for extra in (1.0, 1.5, 0.5):
                    trial = pts[:-1] + [pts[-1] + u * extra]
                    got = acc(trial)
                    if got is not None:
                        pts = trial
                        break
            if stats is not None:
                stats["fine_tried"] = stats.get("fine_tried", 0) + 1
                stats["fine_passed"] = stats.get("fine_passed", 0) + (got is not None)
            if got is None:
                self.bad.add(key)
                continue
            if any(len(fr) == len(pts) and np.allclose(fr, np.asarray(pts)) for fr in self.fragile):
                self.bad.add(key)
                continue
            raw = list(pts)
            pts, got = self.simplify(pts, got, need_red, need_green, sp.simplify if simplify is None else simplify)
            if stats is not None:
                stats.setdefault("corners", []).append((len(raw), len(pts)))
            return pts, got, raw
        return None

    def simplify(self, pts, got, need_red, need_green, how: str):
        def acc(points):
            return self.acceptable(points, need_red, need_green)

        for step in [h for h in how.split("+") if h and h != "none"]:
            if step == "polish":
                pts, got = self.polish(pts, got, need_red, need_green)
            elif step == "shortcut":
                pts, got = simplify_shortcut(pts, acc, got,
                                             lambda t: self.fast_route(t, need_red, need_green) is not None)
            elif step == "rdp":
                pts, got, _ = simplify_rdp(pts, acc, self.sp.rdp_eps, got)
            else:
                raise ValueError(step)
        return pts, got


def _assemble(q, p: RouteParams, li: int, pts, got, need_red: bool, need_green: bool, start_yaw: float) -> RoutePlan:
    """A checked polyline -> RoutePlan, exactly as the tail of safe_route._plan_level does it (that code is not a
    function of its own there; when this module is integrated the two should share one)."""
    from_start = q.start is not None
    n = p.circle_points
    e = None if from_start else q.P[q.entry]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):                     # split long legs
        L = float(np.linalg.norm(b - a))
        m = max(1, int(math.ceil(L / p.max_leg - 1e-9)))
        out += [a + (b - a) * i / m for i in range(1, m + 1)]
    dense, res = got
    waypoints = []
    for i, w in enumerate(out):
        nxt, prv = out[min(i + 1, len(out) - 1)], out[max(i - 1, 0)]
        yaw = math.atan2(nxt[1] - w[1], nxt[0] - w[0]) if i + 1 < len(out) else math.atan2(w[1] - prv[1],
                                                                                         w[0] - prv[0])
        waypoints.append((float(w[0]), float(w[1]), float(wrap(yaw))))
    others = np.array([b.state not in (ENTRY, EXIT) for b in q.buoys])
    clearance = float(np.linalg.norm(dense[:, None, :2] - q.P[None, others], axis=2).min()) if others.any() \
        else math.inf
    margins = [clearance - p.strike_distance, res.min_shield_margin]
    if need_red:
        margins.append(max(ev.credit_margin for ev in res.red_credits))
    if need_green:
        margins.append(max(ev.credit_margin for ev in res.green_credits))
    first, last = out[0], out[-1]
    entry_r, start_angle = 0.0, 0.0
    if not from_start:
        entry_r = min(q.entry_radii, key=lambda R: abs(R - float(np.linalg.norm(first - e))))
        start_angle = math.atan2(first[1] - e[1], first[0] - e[0])
    plan = RoutePlan(
        status="ok", waypoints=waypoints, start_yaw=float(wrap(start_yaw)), poses=dense, check=res,
        passes=[(ev.name or str(ev.index), ev.state, ev.side, round(ev.distance, 2), ev.verdict)
                for ev in res.events],
        red_passed=sorted({ev.name or str(ev.index) for ev in res.red_credits}),
        green_passed=sorted({ev.name or str(ev.index) for ev in res.green_credits}),
        level=li, clearance=clearance, margin=float(min(margins)), length=sr.path_length(waypoints),
        entry_radius=entry_r, entry_start_angle=start_angle, need_red=need_red, need_green=need_green)
    if not from_start:
        loop = sr.circle_goals(e, entry_r, start_angle, 1, n, 0)
        loop[0] = (waypoints[0][0], waypoints[0][1], loop[0][2])
        loop[-1] = waypoints[0]
        plan.entry_circle = loop
    if q.exit_ is not None:
        x = q.P[q.exit_]
        plan.exit_radius = min(q.exit_radii, key=lambda R: abs(R - float(np.linalg.norm(last - x))))
        plan.exit_join_angle = math.atan2(last[1] - x[1], last[0] - x[0])
        ring = sr.circle_goals(x, plan.exit_radius, plan.exit_join_angle, -1, n, p.exit_overshoot_points)
        plan.exit_circle = ring[1:]
    sr._set_goals(plan)
    return plan


# ----------------------------------------------------------------------------------------------------------------
# plan_route
# ----------------------------------------------------------------------------------------------------------------

def plan_route(buoys: Sequence[Buoy], params: Optional[RouteParams] = None, obstacles=None, keep_out=None,
               entry_immune: bool = True, exit_immune: bool = True, start=None, need_red: bool = True,
               need_green: bool = True, *, seed: Optional[int] = None, time_budget: float = 3.0,
               sampler: Optional[SamplerParams] = None, stats: Optional[dict] = None) -> RoutePlan:
    """safe_route.plan_route by random sampling: same arguments, same RoutePlan and statuses.

    seed         None: a different search every call (what a retry wants); an int: the same samples (the result
                 still depends on how many fit in the time unless sampler.max_iterations is set)
    time_budget  s of growing trees, shared between the margin levels. The fine checks, the simplification and the
                 Monte Carlo of each candidate (safe_route.monte_carlo, about as long again) come on top
    stats        a dict that is filled with what the search did
    """
    p = params or RouteParams()
    sp = sampler or SamplerParams()
    st = stats if stats is not None else {}
    rng = np.random.default_rng(seed)
    buoys = list(buoys)
    P, red, green, trusted, confirmed = sr._arrays(buoys)
    entry = next((i for i, b in enumerate(buoys) if b.state == ENTRY), None)
    exit_ = next((i for i, b in enumerate(buoys) if b.state == EXIT), None)
    from_start = start is not None
    if from_start:
        entry_radii = []
    elif entry is None:
        return RoutePlan("missing_entry", "no entry (flashing blue) buoy")
    else:
        entry_radii = [p.entry_radius] if p.entry_radius else sr.radius_candidates(buoys, entry, p)
    exit_radii = ([p.exit_radius] if p.exit_radius else sr.radius_candidates(buoys, exit_, p)) if exit_ is not None \
        else []
    circle_notes = []

    def fewest(index, radii, sigma, what):
        laps = {R: sr.circle_alone(buoys, index, R, sigma, p) for R in radii}
        least = min(len({v.index for v in lap.violations}) for lap in laps.values())
        keep = [R for R in radii if len({v.index for v in laps[R].violations}) == least]
        if least:
            v = {(e.name or str(e.index), e.state, e.side, round(e.distance, 1)) for e in laps[keep[0]].violations}
            circle_notes.append(f"{what} circle (R {keep[0]:.1f}) passes " + ", ".join(
                f"{n} ({s.split('_')[-1]}) to {side} at {d} m" for n, s, side, d in sorted(v)))
        return keep

    if not entry_immune and not from_start:
        entry_radii = fewest(entry, entry_radii, 1, "entry")
    if not exit_immune and exit_ is not None:
        exit_radii = fewest(exit_, exit_radii, -1, "exit")
    tree = cKDTree(np.asarray(obstacles, dtype=float)[:, :2]) if obstacles is not None and len(obstacles) else None
    q = sr._Problem(buoys, P, red, green, trusted, confirmed, entry, exit_, entry_radii, exit_radii, obstacles, tree,
                    keep_out, np.asarray(start, dtype=float)[:2] if from_start else None)
    owed_red, owed_green = need_red, need_green
    has_red, has_green = bool(np.any(red & confirmed)), bool(np.any(green & confirmed))
    want_red, want_green = has_red and owed_red, has_green and owed_green
    notes = []
    if owed_red and not has_red:
        notes.append("no confirmed red to pass")
    if owed_green and not has_green:
        notes.append("no confirmed green to pass")
    if from_start and exit_ is None and not (want_red or want_green):
        return RoutePlan("missing_exit", "; ".join(["no exit (steady blue) buoy and nothing left to pass"] + notes),
                         need_red=False, need_green=False)
    if from_start and exit_ is not None and not (want_red or want_green):
        # already on an exit circle with nothing left to pass: the transit is that one point (as the lattice)
        r = float(np.linalg.norm(q.start - P[exit_]))
        clear = q.clearances(p.levels[0][0], p)
        if any(abs(r - R) <= p.ring_tolerance for R in exit_radii) and np.all(
                np.linalg.norm(P - q.start[None], axis=1) >= clear):
            plan = sr._already_joined(q, p, 0, (float(q.start[0]), float(q.start[1]), 0.0), False, False)
            plan.message = "; ".join(notes + circle_notes)
            return plan

    # the searches: one tree per margin level, the time shared between those still to come
    t0 = time.monotonic()
    deadline = t0 + time_budget
    order = list(sp.levels) if sp.levels is not None else list(range(len(p.levels)))
    all_levels = sp.all_levels and p.choose == "robust"
    asks = [(want_red, want_green)]     # most wanted first: both colours, then one, then just a clean transit
    if want_red and want_green:
        asks += [(True, False), (False, True)]
    if want_red or want_green:
        asks.append((False, False))
    trees, fragile = [], []
    st.update(mc_runs=0, mc_time=0.0)

    def candidate(tr, ask):
        """The tree's best route for the ask as a RoutePlan with its judge, hand-overs and Monte Carlo; None when
        there is none or it is too fragile."""
        if p.mc_trials > 0 and st["mc_runs"] >= sp.max_mc:
            return None
        got = tr.best(ask[0], ask[1], st)
        if got is None:
            return None
        pts, g, _ = got
        c = _assemble(q, p, tr.li, pts, g, ask[0], ask[1], tr.start_yaw_of(pts))
        c.judge = sr.check_plan(c, buoys, p, entry_immune, exit_immune)
        near = [e for e in c.judge.events if e.closest and e.state in (RED, GREEN) and min(
            np.hypot(e.pose[0] - c.waypoints[0][0], e.pose[1] - c.waypoints[0][1]),
            np.hypot(e.pose[0] - c.waypoints[-1][0], e.pose[1] - c.waypoints[-1][1])) <= p.handover_distance]
        c.handover = [str(e) for e in near]
        if p.mc_trials > 0:
            t = time.monotonic()
            c.mc = sr.monte_carlo(c, buoys, p.mc_buoy_sigma, p.mc_trials, seed=1, params=p)
            st["mc_runs"] += 1
            st["mc_time"] += time.monotonic() - t
            if c.mc["full"] < p.min_mc_full:
                fragile.append(c)
                tr.fragile.append(np.asarray(pts))
                return None
        return c

    def choose(candidates):
        if p.mc_trials > 0 and len(candidates) > 1:
            # most often fully compliant; within 2 % of the best: the quieter hand-overs, then the stricter level
            best = max(c.mc["full"] - 0.2 * c.mc["strike"] for c in candidates)
            candidates = [c for c in candidates if c.mc["full"] - 0.2 * c.mc["strike"] >= best - 0.02]
            candidates.sort(key=lambda c: (len(c.handover), c.level))
        return candidates[0]

    full = []
    spent = 0.0                         # search time used so far (the fine checks and Monte Carlo are not counted)
    for n_done, li in enumerate(order):
        if spent >= time_budget:
            break
        tr = _Tree(q, p, li, want_red, want_green, sp, rng)
        trees.append(tr)
        t_level = time.monotonic()
        end = t_level + (time_budget - spent) / (len(order) - n_done)
        validated = None
        while tr.n and tr.n < sp.max_nodes and not (sp.max_iterations and tr.iterations >= sp.max_iterations):
            now = time.monotonic()
            if now >= end:
                break
            if validated is not None and now >= validated + max(sp.improve_min, sp.improve * (validated - t_level)):
                break
            tr.step()
            if validated is None and tr.new_full:
                tr.new_full = False
                if tr.best(want_red, want_green, None, tries=2, simplify="none") is not None:
                    validated = time.monotonic()
        spent += time.monotonic() - t_level
        tr.first = None if validated is None else validated - t_level
        if validated is not None:
            c = candidate(tr, asks[0])
            if c is not None:
                full.append(c)
                if not all_levels:
                    break
    st["search_time"] = spent
    st["nodes"] = [tr.n for tr in trees]
    st["iterations"] = [tr.iterations for tr in trees]
    st["edges"] = [tr.fast.checked for tr in trees]
    st["finishes"] = [len(tr.finishes) for tr in trees]
    st["first"] = [tr.first for tr in trees]
    st["levels"] = [tr.li for tr in trees]
    if st.get("keep_trees"):            # for studies: the trees themselves
        st["trees"], st["problem"] = trees, q

    plan, ask = None, None
    if full:
        plan, ask = choose(full), asks[0]
    else:
        for ask in asks[1:]:
            found = []
            for tr in trees:
                c = candidate(tr, ask)
                if c is not None:
                    found.append(c)
                    if not all_levels:
                        break
            if found:
                plan = choose(found)
                break
    st["fine_calls"] = sum(tr.fine_calls for tr in trees)
    st["total_time"] = time.monotonic() - t0
    if plan is None:
        why = "no compliant transit found at any margin level (sampling planner: not a proof that there is none)"
        if fragile:
            best = max(fragile, key=lambda c: c.mc["full"])
            why = (f"the only transits found are fragile: the best fully complies in {best.mc['full'] * 100:.0f}% "
                   f"of simulated executions (< {p.min_mc_full * 100:.0f}%)")
        return RoutePlan("no_route", "; ".join(notes + circle_notes + [why]), circle_violations=circle_notes)
    need_red, need_green = ask
    if exit_ is None:
        plan.status = "missing_exit"
        notes_ = ["no exit (steady blue) buoy: the route ends after the passes, no exit circle"] + notes
    elif (need_red or not owed_red) and (need_green or not owed_green):
        plan.status, notes_ = "ok", list(notes)
    else:
        plan.status = "ok_partial"
        notes_ = notes + [n for n, have, need in (
            ("no compliant transit passes a red to starboard", want_red, need_red),
            ("no compliant transit passes a green to port", want_green, need_green)) if have and not need]
    if circle_notes and plan.judge.violations:
        plan.status = "circle_violation"
    plan.circle_violations = circle_notes
    plan.message = "; ".join(notes_ + circle_notes)
    if p.merge_legs:
        sr._merge_legs(plan)
    return plan
