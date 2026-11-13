"""Safe Passage geometry on a costmap snapshot, free of ROS so it can be tested offline.

All positions are 2D in the costmap's global frame (asv5/map, ENU). Angles are ENU: radians, anticlockwise from
east. sigma is the circling direction: +1 clockwise seen from above, -1 counterclockwise.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

LETHAL = 100  # nav_msgs/OccupancyGrid scale: 100 lethal, 99 inscribed, -1 unknown

ENTRY, EXIT, RED, GREEN, BLACK = "flashing_blue", "steady_blue", "flashing_red", "flashing_green", "off"
# Circling direction per state: entry and red clockwise (red stays on starboard), green and exit counterclockwise
SIGMA = {ENTRY: 1, RED: 1, GREEN: -1, EXIT: -1}


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


def order_buoys(positions: Sequence, states: Sequence[str], boat) -> list[int]:
    """Circling order: the entry, the reds and greens, the exit. Black buoys are left out.

    Reds and greens are sorted by their projection on the line from the start (the entry, else the boat) to the end
    (the exit, else the reds' and greens' centroid). If start and end coincide they go nearest-next from the start.
    """
    positions = [np.asarray(p, dtype=float)[:2] for p in positions]
    entry = next((i for i, s in enumerate(states) if s == ENTRY), None)
    exit_ = next((i for i, s in enumerate(states) if s == EXIT), None)
    middle = [i for i, s in enumerate(states) if s in (RED, GREEN)]
    start = positions[entry] if entry is not None else np.asarray(boat, dtype=float)[:2]
    if exit_ is not None:
        end = positions[exit_]
    elif middle:
        end = np.mean([positions[i] for i in middle], axis=0)
    else:
        end = start
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length > 1.0:
        middle.sort(key=lambda i: float(np.dot(positions[i] - start, axis / length)))
    else:
        ordered, at = [], start
        while middle:
            nearest = min(middle, key=lambda i: float(np.linalg.norm(positions[i] - at)))
            middle.remove(nearest)
            ordered.append(nearest)
            at = positions[nearest]
        middle = ordered
    return ([entry] if entry is not None else []) + middle + ([exit_] if exit_ is not None else [])


def nearest_other(points) -> np.ndarray:
    """Distance from each point to the nearest other point, inf when alone."""
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    if len(points) < 2:
        return np.full(len(points), np.inf)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def tangent_point(centre, radius: float, start, clockwise: bool) -> np.ndarray:
    """Where a vehicle at start joins the circle tangentially in the given direction.

    From inside or on the circle it joins radially, along its bearing from the centre.
    """
    centre, start = np.asarray(centre, dtype=float)[:2], np.asarray(start, dtype=float)[:2]
    offset = start - centre
    r0 = float(np.linalg.norm(offset))
    phi0 = math.atan2(offset[1], offset[0]) if r0 > 0 else 0.0
    phi = phi0 - (1.0 if clockwise else -1.0) * math.acos(radius / r0) if r0 > radius else phi0
    return centre + radius * np.array([math.cos(phi), math.sin(phi)])


def start_angle(centre, radius: float, boat, sigma: int) -> float:
    """Bearing from centre of the point the boat joins the circle at: the tangent point, or radially from inside."""
    point = tangent_point(centre, radius, boat, sigma > 0) - np.asarray(centre, dtype=float)[:2]
    return math.atan2(point[1], point[0])


def circle_points(centre, radius: float, phi0: float, sigma: int, n: int, overshoot: int = 0) -> list:
    """[(xy, yaw), ...] for k = 0 .. n + overshoot: point k at phi0 - sigma * k * 2 pi / n, yaw along the travel."""
    centre = np.asarray(centre, dtype=float)[:2]
    out = []
    for k in range(n + overshoot + 1):
        phi = phi0 - sigma * k * 2.0 * math.pi / n
        out.append((centre + radius * np.array([math.cos(phi), math.sin(phi)]), phi - sigma * math.pi / 2.0))
    return out


def circle_angles(n: int, offset: float = 0.0) -> np.ndarray:
    return offset + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)


def choose_radius(grid: CostGrid, centre, neighbours, nominal: float, minimum: float, step: float,
                  clearance: float, threshold: int, check_points: int = 16) -> tuple[float, str]:
    """Circle radius around centre and why.

    Starts at nominal, capped at (nearest neighbour - clearance) and floored at minimum, then shrinks by step while
    any of check_points evenly spaced points on the circle costs threshold or more, never below minimum.
    """
    centre = np.asarray(centre, dtype=float)[:2]
    radius, why = nominal, "nominal"
    neighbours = np.atleast_2d(np.asarray(neighbours, dtype=float)) if len(neighbours) else np.zeros((0, 2))
    if len(neighbours):
        nearest = float(np.linalg.norm(neighbours[:, :2] - centre, axis=1).min())
        if nearest - clearance < radius:
            radius, why = nearest - clearance, f"neighbour {nearest:.2f} m"
    if radius < minimum:
        radius, why = minimum, why + ", floored at minimum"

    def blocked(r):
        angles = circle_angles(check_points)
        points = centre + r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return int(grid.cost(points).max()) >= threshold

    while blocked(radius):
        if radius <= minimum + 1e-9:
            return radius, why + ", still blocked at minimum"
        radius = max(radius - step, minimum)
        why += f", shrunk to {radius:.2f} m for cost"
    return radius, why
