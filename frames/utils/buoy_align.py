"""Safe Passage: line the drone's buoy layout up with the costmap. Free of ROS.

The drone's buoy positions can sit metres from the real buoys, but as one layout: a single shift t fixes them all.
The costmap already holds the lidar's accumulated picture, so:

  blobs    lethal cells of the costmap, joined across small gaps; a buoy is a small blob, a pad or dock is not
  solve    every shift on a grid is scored by how close the shifted drone buoys land to their nearest small blob;
           the best one wins if it clearly beats the best different one, then its matches refine it (least squares)
  Aligner  solves each costmap until consecutive solves agree (lock), then keeps t as the median of recent
           solves near the locked one, since a single costmap wobbles with localisation and clears near the boat

Positions are 2D in the costmap frame (asv5/map). Names: C_i drone buoy, t shift, P_i = C_i + t predicted buoy,
E_i estimate, the centre buoy i is circled around.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from frames.utils.safe_passage_geometry import LETHAL, CostGrid, nearest_other

BLOB, PREDICTED = "BLOB", "PREDICTED"


@dataclass
class AlignParams:
    # Blobs
    join_gap: float = 0.6         # m, lethal cells closer than this are one blob (buoys are >= 1.7 m apart)
    blob_min_extent: float = 0.1  # m, a buoy blob's bounding box is at least one cell ...
    blob_max_extent: float = 1.2  # ... and at most this on its longer side (RoboBuoy 0.43 m, smeared)
    field_margin: float = 12.0    # m, the costmap is read this far beyond every position a buoy can shift to
    # Shift search
    max_shift: float = 10.0       # m, shifts searched: +-this in x and y
    shift_step: float = 0.1       # m, grid of the search
    score_sigma: float = 0.4      # m, a shifted buoy this far from a blob scores exp(-1/2); the score is their sum
    min_score: float = 3.5        # accepted from this score (about the number of buoys landing on blobs) ...
    min_ratio: float = 1.5        # ... if it is also this many times the runner-up's
    runner_up_separation: float = 1.0  # m, the runner-up is the best shift at least this far from the winner
    match_radius: float = 0.75    # m, pairs this close refine the winner
    # Lock
    lock_solves: int = 3          # consecutive accepted solves that must agree before the shift is used
    lock_tolerance: float = 0.5   # m, how far apart they may be (1.0 locked up to 1.5 m off while localisation wobbled)
    basin: float = 2.0            # m, once locked only shifts this close to the locked one are searched
    median_window: int = 10       # solves the locked shift is the median of
    # Estimates
    snap_radius: float = 1.0          # m, E_i is the nearest blob this close to P_i, else P_i
    snap_spacing_factor: float = 0.5  # ... and no further than this x the distance to the nearest other drone buoy
    claim_radius: float = 1.0         # m, blobs this close to a buoy already circled are off limits


@dataclass
class Solve:
    t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    score: float = 0.0
    second: float = 0.0           # the runner-up's score
    second_t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    accepted: bool = False
    matched: int = 0              # drone buoys within match_radius of a blob at t


@dataclass
class Estimate:
    xy: np.ndarray                # E_i
    source: str                   # BLOB | PREDICTED
    predicted: np.ndarray         # P_i
    radius: float                 # search radius used around P_i


def blobs(grid: CostGrid, around, reach: float, join_gap: float) -> np.ndarray:
    """Lethal regions of the costmap within reach of the box around the given points: rows (x, y, cells, extent).

    Lethal cells closer than join_gap form one region, so a structure's specks and broken strokes are sized with
    its outline. Extent is the longer side of the region's bounding box in the grid.
    """
    points = np.atleast_2d(np.asarray(around, dtype=float))[:, :2]
    if grid.data.size == 0 or not len(points):
        return np.zeros((0, 4))
    res = grid.resolution
    c, s = math.cos(grid.origin_yaw), math.sin(grid.origin_yaw)
    d = points - [grid.origin_x, grid.origin_y]
    gx, gy = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    pad = reach + join_gap
    col0, col1 = max(int((gx.min() - pad) / res), 0), min(int((gx.max() + pad) / res) + 1, grid.data.shape[1])
    row0, row1 = max(int((gy.min() - pad) / res), 0), min(int((gy.max() + pad) / res) + 1, grid.data.shape[0])
    if col0 >= col1 or row0 >= row1:
        return np.zeros((0, 4))
    lethal = grid.data[row0:row1, col0:col1] >= LETHAL
    grown = lethal
    steps = int(round(join_gap / res / 2.0))
    if steps > 0:
        grown = ndimage.binary_dilation(lethal, structure=np.ones((3, 3), dtype=bool), iterations=steps)
    labels, _ = ndimage.label(grown, structure=np.ones((3, 3)))
    labels = labels * lethal
    out = []
    for k, window in enumerate(ndimage.find_objects(labels), 1):
        if window is None:
            continue
        rows, cols = np.nonzero(labels[window] == k)
        if not len(rows):
            continue
        bx = (col0 + window[1].start + cols.mean() + 0.5) * res
        by = (row0 + window[0].start + rows.mean() + 0.5) * res
        out.append((grid.origin_x + c * bx - s * by, grid.origin_y + s * bx + c * by, len(rows),
                    max(np.ptp(rows) + 1, np.ptp(cols) + 1) * res))
    return np.array(out).reshape(-1, 4)


def solve(candidates: np.ndarray, buoy_blobs: np.ndarray, p: AlignParams, centre=None,
          reach: Optional[float] = None) -> Solve:
    """Best shift of the candidates onto the buoy-sized blobs, searched on a grid of half-width reach about centre.

    Without centre the whole +-max_shift range is searched and the winner must beat the runner-up; with one (a
    locked shift) only the score threshold applies, the neighbourhood being already chosen.
    """
    out = Solve()
    if len(candidates) == 0 or len(buoy_blobs) == 0:
        return out
    local = centre is not None
    centre = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
    reach = p.max_shift if reach is None else reach
    steps = np.arange(-reach, reach + 1e-9, p.shift_step)
    shifts = centre + np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 2)
    tree = cKDTree(buoy_blobs)
    distance, _ = tree.query(candidates[None, :, :] + shifts[:, None, :])
    score = np.exp(-0.5 * (distance / p.score_sigma) ** 2).sum(axis=1)
    best = int(score.argmax())
    t = shifts[best].copy()
    out.score = float(score[best])
    far = np.linalg.norm(shifts - t, axis=1) >= p.runner_up_separation
    if far.any():
        second = int(np.where(far, score, -1.0).argmax())
        out.second, out.second_t = float(score[second]), shifts[second].copy()
    out.accepted = out.score >= p.min_score and (local or out.score >= p.min_ratio * out.second)
    # Least squares on the pairs the winner implies: for a pure shift, the mean residual
    distance, nearest = tree.query(candidates + t)
    pairs = distance <= p.match_radius
    if pairs.sum() >= 3:
        t = t + (buoy_blobs[nearest[pairs]] - (candidates[pairs] + t)).mean(axis=0)
        distance, _ = tree.query(candidates + t)
    out.t, out.matched = t, int((distance <= p.match_radius).sum())
    return out


class Aligner:
    """The shift for one Safe Passage goal. Not thread safe: the caller serialises update() and the readers."""

    def __init__(self, candidates, params: AlignParams):
        self.p = params
        self.candidates = np.atleast_2d(np.asarray(candidates, dtype=float))[:, :2]
        self.nn = nearest_other(self.candidates)
        self.t = np.zeros(2)           # the shift in use: zero until locked
        self.locked = False
        self.last = Solve()            # the latest solve, accepted or not
        self.matched = 0               # drone buoys with a blob at the shift in use
        self.buoy_blobs = np.zeros((0, 2))   # buoy-sized blobs of the latest costmap
        self.structures = np.zeros((0, 4))   # the other blobs, for display
        self._streak: list[np.ndarray] = []
        self._history: list[np.ndarray] = []

    def reach(self) -> float:
        """How far beyond the candidates the costmap has to be read."""
        return self.p.max_shift + self.p.field_margin

    def update(self, all_blobs: np.ndarray) -> Solve:
        """One costmap's blobs (rows x, y, cells, extent): solve, lock when solves agree, refine once locked."""
        p = self.p
        small = (all_blobs[:, 3] >= p.blob_min_extent) & (all_blobs[:, 3] <= p.blob_max_extent)
        self.buoy_blobs, self.structures = all_blobs[small, :2], all_blobs[~small]
        if not self.locked:
            fresh = solve(self.candidates, self.buoy_blobs, p)
            self._streak = self._streak + [fresh.t] if fresh.accepted else []
            recent = self._streak[-p.lock_solves:]
            if len(recent) == p.lock_solves and max(np.linalg.norm(a - recent[-1]) for a in recent) <= p.lock_tolerance:
                self.locked, self._history = True, list(recent)
                self.t = np.median(self._history, axis=0)
        else:
            fresh = solve(self.candidates, self.buoy_blobs, p, self.t, p.basin)
            if fresh.accepted:
                self._history = (self._history + [fresh.t])[-p.median_window:]
                self.t = np.median(self._history, axis=0)
        self.last = fresh
        if len(self.buoy_blobs):
            distance, _ = cKDTree(self.buoy_blobs).query(self.candidates + self.t)
            self.matched = int((distance <= p.match_radius).sum())
        else:
            self.matched = 0
        return fresh

    def predicted(self) -> np.ndarray:
        return self.candidates + self.t

    def search_radius(self, i: int) -> float:
        return min(self.p.snap_radius, self.p.snap_spacing_factor * float(self.nn[i]))

    def estimate(self, i: int, claimed=()) -> Estimate:
        """E_i: the nearest unclaimed buoy-sized blob within the search radius of P_i, else P_i."""
        predicted = self.candidates[i] + self.t
        radius = self.search_radius(i)
        best, best_d = None, math.inf
        for blob in self.buoy_blobs:
            d = float(np.linalg.norm(blob - predicted))
            if d > radius or d >= best_d:
                continue
            if any(np.linalg.norm(blob - np.asarray(c, dtype=float)[:2]) <= self.p.claim_radius for c in claimed):
                continue
            best, best_d = blob, d
        if best is not None:
            return Estimate(best.copy(), BLOB, predicted, radius)
        return Estimate(predicted.copy(), PREDICTED, predicted, radius)
