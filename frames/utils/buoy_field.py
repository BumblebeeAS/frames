"""Safe Passage buoy field from lidar: landmarks, drone-to-lidar shift, per-buoy estimates. Free of ROS.

Positions are 2D in the global frame (asv5/map). Names follow the design:

  C_i  drone candidate i                  P_i = C_i + t  where buoy i is predicted
  L_j  landmark j: a persistent buoy-sized lidar cluster near the candidates
  t    shift that lines the drone's buoy layout up with the landmarks, 0 until enough buoys agree
  r_i  search radius around P_i          E_i  estimate: the centre buoy i is circled around

The field is the union of discs of radius crop_r around every C_i (blacks included): lidar points outside it are
never looked at, so landmarks only exist where the drone saw buoys.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from frames.utils.safe_passage_geometry import CostGrid, lethal_centroid, nearest_other

LANDMARK, COSTMAP, PREDICTED = "LANDMARK", "COSTMAP", "PREDICTED"


@dataclass
class FieldParams:
    field_crop_margin: float = 1.5      # m, crop disc radius beyond max_shift_eff
    cluster_eps: float = 0.3            # m, points closer than this join one cluster
    cluster_min_points: int = 3         # smallest cluster kept
    buoy_min_extent: float = 0.15       # m, accepted 2D size of a buoy cluster (longest side)
    buoy_max_extent: float = 0.9
    edge_margin: float = 0.3            # m, a cluster this close to the crop edge was cut by it
    landmark_assoc_radius: float = 0.5  # m, a detection this close updates a landmark
    landmark_smoothing: float = 0.3     # fraction of the gap each detection moves its landmark
    landmark_min_hits: int = 3          # hits before a landmark is usable
    landmark_expiry: float = 60.0       # s unseen before a landmark is dropped
    max_shift: float = 4.0              # m, largest drone-to-lidar shift considered
    shift_spacing_factor: float = 0.8   # max shift also <= this x the tightest candidate spacing
    shift_inlier_radius: float = 0.75   # m, a shifted candidate this close to a landmark agrees
    shift_min_inliers: int = 3          # agreeing buoys before the shift is used
    snap_radius: float = 2.0            # m, search radius around P_i without an accepted shift
    snap_radius_matched: float = 1.0    # m, with one
    snap_spacing_factor: float = 0.5    # search radius also <= this x the distance to the nearest other candidate
    claim_radius: float = 1.0           # m, landmarks this close to a finished buoy are off limits


@dataclass
class Landmark:
    id: int
    xy: np.ndarray
    hits: int
    first_seen: float
    last_seen: float


@dataclass
class ShiftFit:
    t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    inliers: int = 0          # agreeing candidates, 0 when not accepted
    best_inliers: int = 0     # of the best proposal, accepted or not
    second_inliers: int = 0   # best proposal more than shift_inlier_radius away from it: ambiguity
    accepted: bool = False


@dataclass
class Estimate:
    xy: np.ndarray            # E_i
    source: str               # LANDMARK | COSTMAP | PREDICTED
    predicted: np.ndarray     # P_i
    radius: float             # r_i
    landmark_id: Optional[int] = None


def remove_box(points: np.ndarray, half_x: float, half_y: float) -> np.ndarray:
    """Points outside the axis-aligned box |x| < half_x, |y| < half_y (the boat, in base_link)."""
    inside = (np.abs(points[:, 0]) < half_x) & (np.abs(points[:, 1]) < half_y)
    return points[~inside]


def cluster(xy: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    """Index arrays of the groups of points linked by chains of neighbours closer than eps."""
    n = len(xy)
    if n < max(1, min_points):
        return []
    pairs = cKDTree(xy).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    count, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == k) for k in range(count)]
    return [g for g in groups if len(g) >= min_points]


def extent(xy: np.ndarray) -> float:
    """Longest side of the points' footprint along their principal axes."""
    if len(xy) < 2:
        return 0.0
    centred = xy - xy.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centred.T))
    projected = centred @ vectors
    return float((projected.max(axis=0) - projected.min(axis=0)).max())


def fit_shift(candidates: np.ndarray, landmarks: np.ndarray, max_shift: float, inlier_radius: float,
              min_inliers: int) -> ShiftFit:
    """Shift t lining candidates up with landmarks, by consensus.

    Every candidate-landmark pair within max_shift proposes t = L - C, scored by the candidates with a landmark
    within inlier_radius after shifting. The best (most inliers, then smallest |t|) is refined to the mean residual
    of its inliers and accepted with at least min_inliers.
    """
    fit = ShiftFit()
    if not len(candidates) or not len(landmarks):
        return fit
    tree = cKDTree(landmarks)

    def score(t):
        d, j = tree.query(candidates + t)
        return d <= inlier_radius, j

    proposals = (landmarks[None, :, :] - candidates[:, None, :]).reshape(-1, 2)
    proposals = proposals[np.linalg.norm(proposals, axis=1) <= max_shift]
    if not len(proposals):
        return fit
    scored = sorted(((int(score(t)[0].sum()), -float(np.linalg.norm(t)), i) for i, t in enumerate(proposals)),
                    reverse=True)
    best_count, _, best_index = scored[0]
    best = proposals[best_index]
    fit.best_inliers = best_count
    fit.second_inliers = next((c for c, _, i in scored[1:]
                               if np.linalg.norm(proposals[i] - best) > inlier_radius), 0)
    inliers, matched = score(best)
    refined = (landmarks[matched[inliers]] - candidates[inliers]).mean(axis=0)
    if score(refined)[0].sum() >= best_count:
        best = refined
    if best_count >= min_inliers:
        fit.t, fit.inliers, fit.accepted = best, best_count, True
    return fit


class BuoyField:
    """Landmarks and shift for one Safe Passage goal. Not thread safe: the caller serialises process and estimate."""

    def __init__(self, candidates, params: FieldParams):
        self.p = params
        self.candidates = np.atleast_2d(np.asarray(candidates, dtype=float))[:, :2]
        self.nn = nearest_other(self.candidates)
        tightest = float(self.nn.min()) if len(self.candidates) > 1 else math.inf
        self.max_shift = min(params.max_shift, params.shift_spacing_factor * tightest)
        self.crop_r = self.max_shift + params.field_crop_margin
        self._candidate_tree = cKDTree(self.candidates)
        self.landmarks: list[Landmark] = []
        self._next_id = 0
        self.shift = ShiftFit()

    def usable(self) -> list[Landmark]:
        return [lm for lm in self.landmarks if lm.hits >= self.p.landmark_min_hits]

    def detect(self, xy: np.ndarray) -> tuple[list[np.ndarray], dict]:
        """Buoy-sized cluster centroids among map-frame points, and counts for the log."""
        stats = {"points": len(xy)}
        if not len(xy):
            stats.update(field=0, clusters=0, detections=0)
            return [], stats
        d, _ = self._candidate_tree.query(xy[:, :2])
        inside = d <= self.crop_r
        xy, d = xy[inside, :2], d[inside]
        stats["field"] = len(xy)
        groups = cluster(xy, self.p.cluster_eps, self.p.cluster_min_points)
        stats["clusters"] = len(groups)
        detections = []
        for g in groups:
            size = extent(xy[g])
            if not self.p.buoy_min_extent <= size <= self.p.buoy_max_extent:
                continue
            if d[g].max() > self.crop_r - self.p.edge_margin:
                continue
            detections.append(xy[g].mean(axis=0))
        stats["detections"] = len(detections)
        return detections, stats

    def update(self, detections: list[np.ndarray], now: float) -> dict:
        """Associate detections with landmarks, expire stale ones, refit the shift. Returns what changed."""
        born = 0
        for xy in detections:
            nearest = min(self.landmarks, key=lambda lm: float(np.linalg.norm(lm.xy - xy)), default=None)
            if nearest is not None and np.linalg.norm(nearest.xy - xy) <= self.p.landmark_assoc_radius:
                nearest.xy = nearest.xy + self.p.landmark_smoothing * (xy - nearest.xy)
                nearest.hits += 1
                nearest.last_seen = now
            else:
                self.landmarks.append(Landmark(self._next_id, np.asarray(xy, dtype=float), 1, now, now))
                self._next_id += 1
                born += 1
        before = len(self.landmarks)
        self.landmarks = [lm for lm in self.landmarks if now - lm.last_seen <= self.p.landmark_expiry]
        expired = before - len(self.landmarks)
        previous = self.shift
        usable = self.usable()
        self.shift = fit_shift(self.candidates, np.array([lm.xy for lm in usable]).reshape(-1, 2), self.max_shift,
                               self.p.shift_inlier_radius, self.p.shift_min_inliers)
        changed = (previous.accepted != self.shift.accepted
                   or float(np.linalg.norm(previous.t - self.shift.t)) > 0.3)
        return {"born": born, "expired": expired, "landmarks": len(self.landmarks), "usable": len(usable),
                "shift_changed": changed}

    def search_radius(self, i: int) -> float:
        base = self.p.snap_radius_matched if self.shift.accepted else self.p.snap_radius
        return min(base, self.p.snap_spacing_factor * float(self.nn[i]))

    def estimate(self, i: int, grid: CostGrid, claimed=()) -> Estimate:
        """E_i: nearest unclaimed usable landmark within r_i of P_i, else the costmap's lethal cells there, else P_i."""
        predicted = self.candidates[i] + self.shift.t
        radius = self.search_radius(i)
        claimed = [np.asarray(c, dtype=float)[:2] for c in claimed]
        best, best_d = None, math.inf
        for lm in self.usable():
            d = float(np.linalg.norm(lm.xy - predicted))
            if d > radius or d >= best_d:
                continue
            if any(np.linalg.norm(lm.xy - c) <= self.p.claim_radius for c in claimed):
                continue
            best, best_d = lm, d
        if best is not None:
            return Estimate(best.xy.copy(), LANDMARK, predicted, radius, best.id)
        cells = lethal_centroid(grid, predicted, radius)
        if cells is not None:
            return Estimate(cells, COSTMAP, predicted, radius)
        return Estimate(predicted.copy(), PREDICTED, predicted, radius)
