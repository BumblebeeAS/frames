"""Safe Passage sampling on a costmap snapshot, free of ROS so it can be tested offline.

All positions are 2D in the costmap's global frame (asv5/map, ENU). Angles are ENU: radians, anticlockwise from
east. Every function degrades to the nominal answer on an empty costmap, and none of them fails: when no candidate
is under the threshold, the cheapest one is used anyway.
"""

import math
from dataclasses import dataclass

import numpy as np

LETHAL = 100  # nav_msgs/OccupancyGrid scale: 100 lethal, 99 inscribed, -1 unknown


@dataclass
class CostGrid:
    """Snapshot of a nav_msgs/OccupancyGrid. Unknown and out-of-grid cells cost 0 (open water)."""

    data: np.ndarray  # (height, width) int, row-major from the origin
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    @classmethod
    def empty(cls) -> "CostGrid":
        return cls(np.zeros((0, 0), dtype=np.int16), 1.0, 0.0, 0.0)

    @classmethod
    def from_msg(cls, msg) -> "CostGrid":
        info = msg.info
        q = info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        return cls(data, info.resolution, info.origin.position.x, info.origin.position.y, yaw)

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.atleast_2d(points)[:, :2] - [self.origin_x, self.origin_y]
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        gx = c * d[:, 0] + s * d[:, 1]
        gy = -s * d[:, 0] + c * d[:, 1]
        return np.floor(gx / self.resolution).astype(int), np.floor(gy / self.resolution).astype(int)

    def cost(self, points) -> np.ndarray:
        """Cost at each point, 0 for unknown or outside the grid."""
        col, row = self._cells(np.asarray(points, dtype=float))
        h, w = self.data.shape
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(len(col), dtype=np.int16)
        out[inside] = self.data[row[inside], col[inside]]
        return np.maximum(out, 0)

    def max_cost_along(self, points) -> int:
        points = np.atleast_2d(points)
        return int(self.cost(points).max()) if len(points) else 0

    def lethal_within(self, centre, radius: float) -> np.ndarray:
        """Centres of lethal cells within radius of centre, (n, 2)."""
        rows, cols = np.nonzero(self.data >= LETHAL)
        if not len(rows):
            return np.zeros((0, 2))
        gx = (cols + 0.5) * self.resolution
        gy = (rows + 0.5) * self.resolution
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        xy = np.stack([self.origin_x + c * gx - s * gy, self.origin_y + s * gx + c * gy], axis=1)
        return xy[np.linalg.norm(xy - np.asarray(centre)[:2], axis=1) <= radius]


def segment(a, b, spacing: float) -> np.ndarray:
    """Points from a to b inclusive, at most spacing apart."""
    a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
    n = max(1, math.ceil(np.linalg.norm(b - a) / spacing))
    return a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)


def circle(centre, radius: float, spacing: float) -> np.ndarray:
    n = max(8, math.ceil(2.0 * math.pi * radius / spacing))
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.asarray(centre, dtype=float)[:2] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def snap_centre(grid: CostGrid, nominal, radius: float) -> np.ndarray:
    """Centroid of the lethal cells within radius of nominal (the buoy as the lidar sees it), else nominal."""
    cells = grid.lethal_within(nominal, radius)
    return cells.mean(axis=0) if len(cells) else np.asarray(nominal, dtype=float)[:2]


def choose_radius(grid: CostGrid, centre, radius: float, tol: float, steps: int = 5, spacing: float = 0.2) -> float:
    """Radius in [radius - tol, radius + tol] whose circle has the lowest maximum cost, ties to the nearest radius."""
    candidates = np.linspace(radius - tol, radius + tol, steps) if tol > 0 and steps > 1 else np.array([radius])
    candidates = candidates[candidates > 0]
    return float(min(candidates, key=lambda r: (grid.max_cost_along(circle(centre, r, spacing)), abs(r - radius))))


def tangent_point(centre, radius: float, start, clockwise: bool) -> np.ndarray:
    """Where a vehicle at start joins the circle tangentially in the given direction, as controls' Encircle does.

    From inside or on the circle it joins radially, along its bearing from the centre.
    """
    centre, start = np.asarray(centre, dtype=float)[:2], np.asarray(start, dtype=float)[:2]
    offset = start - centre
    r0 = float(np.linalg.norm(offset))
    phi0 = math.atan2(offset[1], offset[0]) if r0 > 0 else 0.0
    phi = phi0 - (1.0 if clockwise else -1.0) * math.acos(radius / r0) if r0 > radius else phi0
    return centre + radius * np.array([math.cos(phi), math.sin(phi)])


def _near_any(point: np.ndarray, others, radius: float) -> bool:
    return any(np.linalg.norm(point - np.asarray(o, dtype=float)[:2]) <= radius for o in others)


def choose_approach(grid: CostGrid, centre, radius: float, offset: float, boat, clockwise: bool,
                    threshold: int, step_deg: float = 30.0, spacing: float = 0.2,
                    exclude=(), exclude_radius: float = 1.0) -> np.ndarray:
    """Point on the radius + offset ring to drive to before circling.

    Candidates go round from the boat's bearing to the centre, nearest first. Each is scored on its own cost and the
    straight run Encircle then makes to its tangent point; the first under threshold wins, else the cheapest.
    Candidates within exclude_radius of an exclude point (failed targets) are skipped, unless that skips them all.
    """
    centre, boat = np.asarray(centre, dtype=float)[:2], np.asarray(boat, dtype=float)[:2]
    ring = radius + offset
    bearing = math.atan2(*(boat - centre)[::-1]) if np.linalg.norm(boat - centre) > 0 else 0.0
    steps = max(1, round(360.0 / step_deg))
    deltas = sorted((math.radians(step_deg) * k for k in range(-(steps // 2), steps - steps // 2)), key=abs)
    points = [centre + ring * np.array([math.cos(bearing + d), math.sin(bearing + d)]) for d in deltas]
    allowed = [(d, p) for d, p in zip(deltas, points) if not _near_any(p, exclude, exclude_radius)]
    scored = []
    for delta, point in allowed or list(zip(deltas, points)):
        run = segment(point, tangent_point(centre, radius, point, clockwise), spacing)
        score = max(grid.max_cost_along(point), grid.max_cost_along(run))
        if score < threshold:
            return point
        scored.append((score, abs(delta), point))
    return min(scored, key=lambda item: item[:2])[2]


def sample_point(grid: CostGrid, nominal, radius: float, rings: int = 2, per_ring: int = 8,
                 ahead_of=None) -> np.ndarray:
    """Cheapest of nominal and rings of candidates within radius, ties to the nearest to nominal.

    Keep radius below the standoff distance so a standoff sample stays on its side of the buoy.
    ahead_of=(point, direction, min_gap) keeps only candidates at least min_gap past point along the unit direction,
    unless none are.
    """
    nominal = np.asarray(nominal, dtype=float)[:2]
    candidates = [nominal]
    for k in range(1, rings + 1):
        r = radius * k / rings
        candidates += [nominal + r * np.array([math.cos(a), math.sin(a)])
                       for a in np.linspace(0.0, 2.0 * math.pi, per_ring, endpoint=False)]
    if ahead_of is not None:
        point, direction, min_gap = ahead_of
        point, direction = np.asarray(point, dtype=float)[:2], np.asarray(direction, dtype=float)[:2]
        # 1e-9: the worst case lands exactly on min_gap
        candidates = [c for c in candidates if np.dot(c - point, direction) >= min_gap - 1e-9] or candidates
    costs = grid.cost(np.array(candidates))
    best = min(range(len(candidates)), key=lambda i: (costs[i], np.linalg.norm(candidates[i] - nominal)))
    return candidates[best]


def crossing_direction(previous, following) -> np.ndarray:
    """Unit direction to cross a pass-through point: from the point before it to the point after it."""
    d = np.asarray(following, dtype=float)[:2] - np.asarray(previous, dtype=float)[:2]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else np.array([1.0, 0.0])


def circle_turn_deg(clockwise: bool, overshoot_deg: float) -> float:
    """One full round plus overshoot, in controls' Encircle convention: positive clockwise seen from above."""
    return (1.0 if clockwise else -1.0) * (360.0 + overshoot_deg)
