"""Safe Passage geometry on a costmap snapshot, free of ROS so it can be tested offline.

All positions are 2D in the costmap's global frame (asv5/map, ENU). The route itself is frames.utils.safe_route.
"""

import math
from dataclasses import dataclass

import numpy as np

LETHAL = 100  # nav_msgs/OccupancyGrid scale: 100 lethal, 99 inscribed, -1 unknown

ENTRY, EXIT, RED, GREEN, BLACK = "flashing_blue", "steady_blue", "flashing_red", "flashing_green", "off"


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

    def lethal_points(self) -> np.ndarray:
        """(N, 2) centres of the lethal cells."""
        rows, cols = np.nonzero(self.data >= LETHAL)
        gx, gy = (cols + 0.5) * self.resolution, (rows + 0.5) * self.resolution
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return np.stack([self.origin_x + c * gx - s * gy, self.origin_y + s * gx + c * gy], axis=1)


def nearest_other(points) -> np.ndarray:
    """Distance from each point to the nearest other point, inf when alone."""
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    if len(points) < 2:
        return np.full(len(points), np.inf)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)
