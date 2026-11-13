"""Safe Passage planner and pass-rule checker (RobotX 2026 Task 1). Free of ROS: numpy + scipy only.

The run is: entry loop (clockwise, full) -> transit -> exit circle (counterclockwise, full).

What is judged, by default (both circles immune):
  from  the point where the entry loop closes: the loop begins at a chosen point, goes round the entry buoy clockwise
        and returns to that point. The loop and the approach before it are immune. The turn the boat makes at the
        closing point, and everything after it, is judged. Nothing is driven past the closing point along the circle.
  to    the point where the boat joins the exit circle. From then on it is immune.
On the judged path the boat must
  (a) never pass a red buoy to port or a green buoy to starboard,
  (b) pass at least one red to starboard and at least one green to port,
  (c) keep clear of every buoy and obstacle.
plan_route(entry_immune=False / exit_immune=False) and check_plan() judge that circle's lap as well.

Pass rule (handbook: "A buoy is considered passed if it is the closest buoy on that side of the vessel when it is
directly abeam"), as modelled here for a vessel pose (x, y, yaw):
  along_b = (b - p) . (cos yaw, sin yaw)      lat_b = (b - p) . (-sin yaw, cos yaw)      (lat > 0: port)
  buoy b comes ABEAM when along_b changes sign, whether the boat drives past it or turns on the spot (ASV5 turns on
  the spot at every waypoint, and a turn sweeps the beam over buoys just like driving does);
  its SIDE is the sign of lat_b at that moment; "that side of the vessel" is the whole half plane on that side;
  it is PASSED if no other buoy (of any state: blue and black buoys count) on that half plane is nearer.
A red passed to port or a green passed to starboard is a violation; a red passed to starboard / a green passed to
port is a credit.

Two readings of the rule are used:
  judge()   the literal rule, no margins: what a referee applying the handbook sentence would conclude. Used on
            actual tracks and in the Monte Carlo.
  strict    the planning rule, with margins on everything that is uncertain: a wrong-side buoy abeam is a violation
            unless another TRUSTED buoy on that side is nearer by shield_margin; within near_range a wrong-side
            buoy anywhere on the beam is a violation whatever shields it; a credit counts only when the buoy is
            nearer than every other object on that side by credit_margin and within credit_range.

Frames: 2D map frame (asv5/map, ENU), yaw in rad anticlockwise from east. Port is left of the heading.

API
  plan_route(buoys, params=RouteParams(), obstacles=None, keep_out=None, entry_immune=True, exit_immune=True,
             start=None, need_red=True, need_green=True)
      -> RoutePlan                                               circles + transit, end to end; with start (x, y):
                                                                 no entry loop, the transit from there, looking only
                                                                 for the passes still needed
  check_plan(plan, buoys, params, entry_immune=True, exit_immune=True) -> CheckResult   literal rule on the plan
  check_trajectory(poses, buoys, params) -> CheckResult          poses: (N, 3) x, y, yaw; any track or plan
  PassTracker(buoys, params).update(pose) -> [PassEvent]         the same checker, one pose at a time (runtime)
  check_path(xy, start_yaw, buoys, params)                       a tracker plan (/asv5/plan) driven ASV5's way
  densify(waypoints, start_yaw, params) -> (N, 3)                ASV5 motion model: turn on the spot, drive
  circle_alone(buoys, index, radius, sigma, params)              what one full circle passes by the literal rule
  simulate_execution / monte_carlo                               executed-route robustness
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ENTRY, EXIT, RED, GREEN, BLACK = "flashing_blue", "steady_blue", "flashing_red", "flashing_green", "off"
UNKNOWN = "unknown"   # a costmap object the drone did not report: obstacle, maybe a buoy, never trusted as a shield
PORT, STARBOARD = "port", "starboard"
WRONG_SIDE = {RED: PORT, GREEN: STARBOARD}      # passing a red to port / a green to starboard is the violation
RIGHT_SIDE = {RED: STARBOARD, GREEN: PORT}


@dataclass
class Buoy:
    x: float
    y: float
    state: str                 # ENTRY, EXIT, RED, GREEN, BLACK or UNKNOWN
    confirmed: bool = True     # it sits on a costmap blob (a drone buoy without one may be a false detection or
    #                            metres off): only confirmed buoys earn credit or act as shields
    sigma: float = 0.3         # m, position uncertainty (1 sd). Adds clearance_sigma_gain x sigma to its clearance
    name: str = ""

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class RouteParams:
    # --- the rule -------------------------------------------------------------------------------------------------
    violation_range: float = 40.0     # m, a wrong-side buoy abeam further than this is ignored (the handbook gives
    #                                   no limit, so this spans the whole field)
    credit_range: float = 6.0         # m, a pass only counts as a credit when the buoy is this close (a referee
    #                                   has to see it as a pass)
    shield_margin: float = 1.0        # m, strict: a wrong-side buoy abeam is harmless only if a trusted buoy on that
    #                                   side is nearer by at least this
    shield_lat_min: float = 0.5       # m, ... and that shield is at least this far onto that side
    credit_margin: float = 0.75       # m, strict: a credited buoy is nearer than every other object on its side by
    #                                   at least this
    credit_lat_slack: float = 0.5     # m, objects within this of the centreline count as "on that side" for that test
    near_range: float = 5.0           # m, strict: this close, a red anywhere on the port beam (or a green on the
    #                                   starboard beam) is a violation whether or not another buoy is nearer: a
    #                                   referee watching would call it passed on the wrong side
    near_sector_deg: float = 60.0     # deg, "on the beam" for that: within this of abeam
    count_reverse_abeam: bool = True  # a buoy coming abeam from astern (turning or reversing) counts like any other
    abeam_hysteresis_deg: float = 1.0  # deg, a buoy must go this far past abeam for the event to register (noise)
    literal: bool = False             # True: no margins at all, every buoy counts and shields (see judge())
    # --- clearance ------------------------------------------------------------------------------------------------
    clearance: float = 2.0            # m, boat centre to buoy centre. Costmap: buoy mark ~0.3 m + inflation 1.25 m,
    #                                   so >= ~1.6 m is zero-cost water where the tracker plans a straight line
    clearance_sigma_gain: float = 1.0  # extra clearance per metre of a buoy's sigma beyond 0.3 m
    obstacle_clearance: float = 1.6   # m, to obstacle points (lethal costmap cells that are not buoys): robot
    #                                   radius 0.75 + inflation, as for buoys
    strike_distance: float = 0.9      # m, boat centre to buoy centre counted as a strike (half beam 0.45 + buoy)
    # --- circles --------------------------------------------------------------------------------------------------
    entry_radius: Optional[float] = None  # m, fix the entry circle's radius; None: chosen from radius_candidates()
    exit_radius: Optional[float] = None   # m, same for the exit circle
    circle_radius: float = 2.5        # m, nominal, as in asv5_safe_passage.yaml
    circle_radius_min: float = 1.6
    neighbour_clearance: float = 1.25  # m, R <= distance to the nearest other buoy - this
    circle_radii: tuple = (1.6, 2.0, 2.5, 3.0)   # m, radii offered besides the node's own
    circle_points: int = 6            # goals per lap, as the node drives it (a hexagon)
    exit_overshoot_points: int = 1    # goals past 360 deg on the exit circle (immune, so the loop surely closes).
    #                                   The entry loop has none: anything past its closing point is judged
    ring_tolerance: float = 0.3       # m, lattice cells this close to a circle are on it
    handover_distance: float = 1.5    # m, passes this close to a hand-over are reported as ambiguous
    # --- search ---------------------------------------------------------------------------------------------------
    grid_step: float = 0.5            # m
    grid_margin: float = 8.0          # m of water searched beyond the outermost buoy
    heading_tolerance_deg: float = 14.0  # deg, strict: a pose is forbidden when an unshielded wrong-side buoy is
    #                                   within this of abeam (covers heading vs travel, measured sd 4-7 deg, 99% 17,
    #                                   and the lattice's 27 deg heading steps)
    max_turn_deg: float = 67.5        # deg, largest turn at one corner: no doubling back to collect a pass
    double_back_deg: float = 120.0    # deg, turning one way by more than this ...
    double_back_window: float = 4.0   # m, ... within this much path is doubling back: not allowed
    corner_cost: float = 1.0          # m of path one corner (a stop, a turn, a start) is worth
    turn_cost: float = 1.0            # m of path one radian of turning on the spot is worth
    soft_clearance: float = 3.5       # m, closer than this to a buoy costs extra ...
    soft_weight: float = 2.0          # ... up to this many times the path length at the hard clearance
    # margin ladder: (clearance, shield_margin, credit_margin, near_range), strictest first
    levels: tuple = ((2.5, 1.5, 1.0, 5.0), (2.0, 1.5, 1.0, 4.0), (1.6, 1.5, 0.75, 3.0), (1.6, 1.0, 0.5, 2.5))
    choose: str = "robust"            # "robust": plan at every level and keep the route that most often complies in
    #                                   a Monte Carlo of its execution; "strictest": the first level with a route
    mc_trials: int = 150              # trials per candidate for "robust"
    mc_buoy_sigma: float = 0.5        # m, buoy position error used there
    min_mc_full: float = 0.7          # a route that fully complies in fewer of those executions than this is dropped
    search_retries: int = 6           # times the search is rerun around a spot where its route fails the fine check
    debug: bool = False               # print why levels fail
    # --- waypoints ------------------------------------------------------------------------------------------------
    max_leg: float = 3.0              # m, longer legs are split: on legs <= 3.5 m the tracked path stayed within
    #                                   0.45 m (90%) of the straight line, on longer ones up to 2.5 m
    merge_legs: bool = True           # the plan is made and chosen with legs split at max_leg, then handed out with
    #                                   one goal per corner (replayed through the real planner: 98 % against 99 %
    #                                   compliant, a quarter less time)
    start_snap: float = 1.5           # m, a route from a given start begins on the nearest allowed lattice cell
    #                                   within this of it (the boat may be stopped inside a buoy's clearance)
    min_leg: float = 0.8              # m, corners closer than this are merged when the rule still holds
    sample_step: float = 0.1          # m, spacing of the dense pose trajectory
    turn_step_deg: float = 2.0        # deg, same for turns on the spot

    def judge(self) -> "RouteParams":
        """The literal handbook rule: no margins, no range limits, every buoy counts."""
        return replace(self, literal=True, violation_range=math.inf, credit_range=math.inf, shield_margin=0.0,
                       shield_lat_min=0.0, credit_margin=0.0, credit_lat_slack=0.0, near_range=0.0)


@dataclass
class PassEvent:
    index: int                 # buoy index in the list given
    name: str
    state: str
    side: str                  # PORT or STARBOARD
    distance: float            # m, boat to buoy when abeam
    step: int                  # index of the pose at which it registered
    pose: tuple                # (x, y, yaw) there
    closest: bool              # literally the closest buoy on that side
    shield_margin: float       # m, how much nearer the nearest trusted buoy on that side is (+: shielded by that much)
    credit_margin: float       # m, how much further the nearest other object on that side is (+: clearly the closest)
    verdict: str               # "credit" | "violation" | "shielded" (wrong side, not the closest) | "neutral"
    reverse: bool = False      # it came abeam from astern (turn or reversing)
    kind: str = "abeam"        # "abeam", or "near": strict only, a wrong-side buoy on the beam within near_range

    def __str__(self) -> str:
        how = "abeam" if self.kind == "abeam" else "close on the beam"
        return (f"{self.name or self.index}:{self.state} {how} to {self.side} at {self.distance:.1f} m -> "
                f"{self.verdict}" + (" (from astern)" if self.reverse else ""))


@dataclass
class CheckResult:
    events: list
    violations: list           # events with verdict "violation"
    red_credits: list          # reds passed to starboard
    green_credits: list        # greens passed to port
    min_distance: float        # m, closest approach to any buoy
    closest_buoy: int
    min_shield_margin: float   # m, over wrong-side events that were not violations (inf when none)
    min_credit_margin: float   # m, best credit margin of each colour, the smaller of the two (-inf when one is missing)
    strikes: list              # buoy indices closer than strike_distance

    @property
    def compliant(self) -> bool:
        return not self.violations

    def satisfied(self, need_red: bool = True, need_green: bool = True) -> bool:
        return (not self.violations and (bool(self.red_credits) or not need_red)
                and (bool(self.green_credits) or not need_green))

    def summary(self) -> str:
        def names(evs):
            return ",".join(str(e.name or e.index) for e in evs) or "-"
        return (f"violations {len(self.violations)} [{names(self.violations)}]; reds to starboard "
                f"[{names(self.red_credits)}]; greens to port [{names(self.green_credits)}]; closest approach "
                f"{self.min_distance:.2f} m")


@dataclass
class RoutePlan:
    status: str                # "ok" | "ok_partial" | "no_route" | "missing_entry" | "missing_exit" | "circle_violation"
    message: str = ""
    # the transit, which is what is judged
    waypoints: list = field(default_factory=list)   # [(x, y, yaw)]: waypoints[0] is where the entry loop begins and
    #                            closes; each later one is one TrackPath goal; the last is the join point on the
    #                            exit circle. yaw = direction to the next one (the last: the direction it arrives in)
    start_yaw: float = 0.0     # rad, heading the boat has as the entry loop closes (arriving along the last chord)
    poses: Optional[np.ndarray] = None              # (N, 3) planned judged trajectory: turn at waypoints[0], legs
    check: Optional[CheckResult] = None             # strict check of poses
    judge: Optional[CheckResult] = None             # literal check of everything that is judged (check_plan)
    passes: list = field(default_factory=list)      # [(name, state, side, distance, verdict)] in order (strict)
    red_passed: list = field(default_factory=list)  # names of reds credited to starboard
    green_passed: list = field(default_factory=list)
    handover: list = field(default_factory=list)    # reds/greens that are abeam as the closest on their side within
    #                            handover_distance of either hand-over: the ambiguous passes (best empty)
    level: int = -1            # index into params.levels that produced the route
    clearance: float = 0.0     # m, closest approach of the transit to any buoy other than the two circled ones
    margin: float = 0.0        # m, the smallest of: clearance - strike_distance, shield margins, credit margins
    length: float = 0.0        # m, of the transit
    # the circles
    entry_radius: float = 0.0
    entry_start_angle: float = 0.0   # rad, bearing from the entry buoy of the point the loop begins and closes at
    entry_circle: list = field(default_factory=list)   # [(x, y, yaw)] goals: [0] the start point (the approach to it
    #                            is free), then round clockwise, the last back on the start point facing the transit
    exit_radius: float = 0.0
    exit_join_angle: float = 0.0     # rad, bearing from the exit buoy of the join point (= waypoints[-1])
    exit_circle: list = field(default_factory=list)    # [(x, y, yaw)] goals after the join point, counterclockwise
    goals: list = field(default_factory=list)       # [(x, y, yaw, phase)] everything in order, phase "entry" |
    #                            "transit" | "exit": one TrackPath goal each
    circle_violations: list = field(default_factory=list)  # for circles that are not immune
    need_red: bool = True      # what the transit set out to earn
    need_green: bool = True
    mc: Optional[dict] = None  # monte_carlo() of this route when several levels competed (choose = "robust")

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_partial")


def circle_radius(buoys: Sequence[Buoy], index: int, p: RouteParams) -> float:
    """The radius safe_passage_node circles buoy `index` with, before any shrinking for costmap cost."""
    c = buoys[index].xy
    others = [float(np.linalg.norm(b.xy - c)) for i, b in enumerate(buoys) if i != index]
    nearest = min(others, default=math.inf)
    return float(max(min(p.circle_radius, nearest - p.neighbour_clearance), p.circle_radius_min))


def wrap(a):
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def _arrays(buoys: Sequence[Buoy]):
    P = np.array([[b.x, b.y] for b in buoys], dtype=float).reshape(-1, 2)
    red = np.array([b.state == RED for b in buoys], dtype=bool)
    green = np.array([b.state == GREEN for b in buoys], dtype=bool)
    trusted = np.array([b.confirmed and b.state != UNKNOWN for b in buoys], dtype=bool)
    confirmed = np.array([b.confirmed for b in buoys], dtype=bool)
    return P, red, green, trusted, confirmed


# ----------------------------------------------------------------------------------------------------------------
# The checker
# ----------------------------------------------------------------------------------------------------------------

class PassTracker:
    """The pass rule, fed one pose at a time. Use on a planned trajectory, a tracker plan, or the live pose."""

    def __init__(self, buoys: Sequence[Buoy], params: Optional[RouteParams] = None):
        self.buoys = list(buoys)
        self.p = params or RouteParams()
        self.P, self.red, self.green, self.trusted, self.confirmed = _arrays(self.buoys)
        if self.p.literal:
            self.trusted = np.ones(len(self.buoys), dtype=bool)
            self.confirmed = np.ones(len(self.buoys), dtype=bool)
        self.sign = np.zeros(len(self.buoys), dtype=int)   # +1 ahead of the beam, -1 abaft, 0 not yet known
        self.step = -1
        self.events: list[PassEvent] = []
        self.min_distance = math.inf
        self.closest_buoy = -1
        self._sin_h = math.sin(math.radians(self.p.abeam_hysteresis_deg))
        self._cos_near = math.cos(math.radians(self.p.near_sector_deg))
        self._near = np.zeros(len(self.buoys), dtype=bool)  # wrong-side buoys now close on the beam

    def classify(self, b: int, x: float, y: float, yaw: float) -> PassEvent:
        """What buoy b being abeam at this pose amounts to."""
        p = self.p
        d = self.P - [x, y]
        lat = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
        dist = np.hypot(d[:, 0], d[:, 1])
        s = 1.0 if lat[b] > 0 else -1.0
        side = PORT if s > 0 else STARBOARD
        others = np.ones(len(dist), dtype=bool)
        others[b] = False
        same = others & (lat * s > 0)
        closest = not np.any(dist[same] < dist[b])
        shield = others & self.trusted & (lat * s >= p.shield_lat_min)
        shield_margin = float(dist[b] - dist[shield].min()) if shield.any() else -math.inf
        rival = others & (lat * s > -p.credit_lat_slack)
        credit_margin = float(dist[rival].min() - dist[b]) if rival.any() else math.inf
        state = self.buoys[b].state
        verdict = "neutral"
        if WRONG_SIDE.get(state) == side:
            if p.literal:
                verdict = "violation" if closest else "shielded"
            elif dist[b] <= p.violation_range:
                verdict = "shielded" if shield_margin >= p.shield_margin and dist[b] > p.near_range else "violation"
        elif RIGHT_SIDE.get(state) == side:
            if p.literal:
                verdict = "credit" if closest else "neutral"
            elif self.confirmed[b] and dist[b] <= p.credit_range and credit_margin >= p.credit_margin:
                verdict = "credit"
        return PassEvent(b, self.buoys[b].name, state, side, float(dist[b]), self.step, (x, y, yaw), closest,
                         shield_margin, credit_margin, verdict)

    def update(self, pose) -> list:
        """Next pose (x, y, yaw). Returns the pass events it caused."""
        self.step += 1
        if not len(self.buoys):
            return []
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        d = self.P - [x, y]
        along = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        dist = np.hypot(d[:, 0], d[:, 1])
        k = int(dist.argmin())
        if dist[k] < self.min_distance:
            self.min_distance, self.closest_buoy = float(dist[k]), k
        band = dist * self._sin_h
        new = np.where(along > band, 1, np.where(along < -band, -1, self.sign))
        out = []
        for b in np.nonzero((new != self.sign) & (self.sign != 0))[0]:
            reverse = new[b] > 0
            if reverse and not self.p.count_reverse_abeam:
                continue
            e = self.classify(int(b), x, y, yaw)
            e.reverse = bool(reverse)
            out.append(e)
        self.sign = new
        if not self.p.literal and self.p.near_range > 0:
            lat = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
            wrong = ((lat > 0) & self.red) | ((lat < 0) & self.green)
            near = wrong & (dist <= self.p.near_range) & (np.abs(along) <= self._cos_near * dist)
            for b in np.nonzero(near & ~self._near)[0]:
                if any(e.index == b and e.step == self.step for e in out):
                    continue
                e = self.classify(int(b), x, y, yaw)
                e.verdict, e.kind = "violation", "near"
                out.append(e)
            self._near = near
        self.events += out
        return out

    def result(self) -> CheckResult:
        ev = self.events
        violations = [e for e in ev if e.verdict == "violation"]
        reds = [e for e in ev if e.verdict == "credit" and e.state == RED]
        greens = [e for e in ev if e.verdict == "credit" and e.state == GREEN]
        shields = [e.shield_margin for e in ev if e.verdict == "shielded"]
        cm = min(max((e.credit_margin for e in reds), default=-math.inf),
                 max((e.credit_margin for e in greens), default=-math.inf))
        strikes = []
        return CheckResult(ev, violations, reds, greens, self.min_distance, self.closest_buoy,
                           min(shields, default=math.inf), cm, strikes)


def check_trajectory(poses, buoys: Sequence[Buoy], params: Optional[RouteParams] = None,
                     yaw_offset: float = 0.0) -> CheckResult:
    """The pass rule along a pose trajectory (N, 3): x, y, yaw. Slice the trajectory to the transit before calling:
    the entry and exit circles are not part of it. yaw_offset turns every heading by that much (robustness)."""
    params = params or RouteParams()
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    tracker = PassTracker(buoys, params)
    for x, y, yaw in poses:
        tracker.update((x, y, yaw + yaw_offset))
    res = tracker.result()
    if len(buoys) and len(poses):
        P = tracker.P
        d = np.linalg.norm(poses[:, None, :2] - P[None], axis=2).min(axis=0)
        res.strikes = [int(i) for i in np.nonzero(d < params.strike_distance)[0]]
    return res


def densify(waypoints, start_yaw: Optional[float] = None, params: Optional[RouteParams] = None,
            end_yaw: Optional[float] = None) -> np.ndarray:
    """ASV5's motion through TrackPath goals: turn on the spot (the shorter way) to face the next waypoint, drive
    straight to it; with end_yaw, turn to it at the last one. waypoints: [(x, y, ...)], any yaw in them is ignored.
    start_yaw None: the trajectory starts already facing the first leg."""
    params = params or RouteParams()
    W = [np.asarray(w, dtype=float) for w in waypoints]
    dth = math.radians(params.turn_step_deg)
    out = []
    yaw = start_yaw

    def turn(x, y, a, b):
        delta = float(wrap(b - a))
        n = max(1, int(math.ceil(abs(delta) / dth)))
        return [(x, y, a + delta * i / n) for i in range(1, n + 1)]

    for a, b in zip(W[:-1], W[1:]):
        leg = b[:2] - a[:2]
        L = float(np.linalg.norm(leg))
        if L < 1e-6:
            continue
        direction = math.atan2(leg[1], leg[0])
        if yaw is None:
            yaw = direction
            out.append((a[0], a[1], yaw))
        elif not out:
            out.append((a[0], a[1], yaw))
        out += turn(a[0], a[1], yaw, direction)
        yaw = direction
        n = max(1, int(math.ceil(L / params.sample_step)))
        out += [(a[0] + leg[0] * i / n, a[1] + leg[1] * i / n, yaw) for i in range(1, n + 1)]
    if end_yaw is not None and out:
        out += turn(out[-1][0], out[-1][1], yaw, end_yaw)
    return np.array(out, dtype=float).reshape(-1, 3)


def check_path(xy, start_yaw: float, buoys: Sequence[Buoy], params: Optional[RouteParams] = None,
               end_yaw: Optional[float] = None) -> CheckResult:
    """A tracker plan (/asv5/plan or simplified_path positions) driven the way ASV5 drives it, against the rule."""
    return check_trajectory(densify(np.asarray(xy, dtype=float)[:, :2], start_yaw, params, end_yaw), buoys, params)


def path_length(waypoints) -> float:
    W = np.asarray([w[:2] for w in waypoints], dtype=float)
    return float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum()) if len(W) > 1 else 0.0


# ----------------------------------------------------------------------------------------------------------------
# The planner
# ----------------------------------------------------------------------------------------------------------------

# 16 headings as lattice moves (di, dj); the heading is the direction of the move
_MOVES = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
          (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]
_K = len(_MOVES)
_YAW = np.array([math.atan2(dj, di) for di, dj in _MOVES])


def circle_goals(centre, radius: float, phi0: float, sigma: int, n: int, overshoot: int = 0) -> list:
    """[(x, y, yaw)] for k = 0 .. n + overshoot, point k at bearing phi0 - sigma k 2 pi / n from the centre, yaw
    along the circle (sigma +1 clockwise). The same points safe_passage_geometry.circle_points gives."""
    out = []
    for k in range(n + overshoot + 1):
        phi = phi0 - sigma * k * 2.0 * math.pi / n
        out.append((centre[0] + radius * math.cos(phi), centre[1] + radius * math.sin(phi),
                    float(wrap(phi - sigma * math.pi / 2.0))))
    return out


def arrival_yaw(phi: float, sigma: int, n: int) -> float:
    """Heading of a boat arriving at the circle point at bearing phi along the polygon's last chord."""
    return phi - sigma * math.pi / 2.0 + sigma * math.pi / n


def radius_candidates(buoys: Sequence[Buoy], index: int, p: RouteParams) -> list:
    """Radii a buoy may be circled with: the node's own, and others that keep neighbour_clearance from every
    other buoy, largest first."""
    c = buoys[index].xy
    nearest = min((float(np.linalg.norm(b.xy - c)) for i, b in enumerate(buoys) if i != index), default=math.inf)
    out = [circle_radius(buoys, index, p)]
    for r in p.circle_radii:
        if p.circle_radius_min - 1e-9 <= r <= nearest - p.neighbour_clearance and all(abs(r - o) > 0.45 for o in out):
            out.append(float(r))
    return sorted(out, reverse=True)


def circle_alone(buoys: Sequence[Buoy], index: int, radius: float, sigma: int, p: RouteParams,
                 phi0: float = 0.0, polygon: bool = True) -> CheckResult:
    """What one full circle of buoy `index` passes by the literal rule (sigma +1 clockwise, -1 counterclockwise)."""
    c = buoys[index].xy
    if polygon:
        pts = circle_goals(c, radius, phi0, sigma, p.circle_points, 1)
        poses = densify(pts, None, p)
    else:
        a = phi0 - sigma * np.linspace(0.0, 2.0 * math.pi + 0.2, int(2.0 * math.pi * radius / 0.05) + 2)
        poses = np.stack([c[0] + radius * np.cos(a), c[1] + radius * np.sin(a), a - sigma * math.pi / 2.0], axis=1)
    return check_trajectory(poses, buoys, p.judge())


def _buoy_clearance(buoys, clearance, p: RouteParams, entry_r: float, exit_r: float) -> np.ndarray:
    out = []
    for b in buoys:
        c = clearance + p.clearance_sigma_gain * max(0.0, b.sigma - 0.3)
        if b.state == ENTRY:
            c = min(c, entry_r - p.ring_tolerance - 0.05)
        elif b.state == EXIT:
            c = min(c, exit_r - p.ring_tolerance - 0.05)
        out.append(c)
    return np.array(out)


@dataclass
class _Problem:
    buoys: list
    P: np.ndarray
    red: np.ndarray
    green: np.ndarray
    trusted: np.ndarray
    confirmed: np.ndarray
    entry: Optional[int]
    exit_: Optional[int]
    entry_radii: list
    exit_radii: list
    obstacles: Optional[np.ndarray]
    tree: Optional[cKDTree]
    keep_out: object
    start: Optional[np.ndarray] = None    # the transit begins here instead of where an entry loop closes

    def clearances(self, clearance: float, p: RouteParams) -> np.ndarray:
        """Per buoy. From a given start nothing is circled at the entry buoy, so it keeps its full clearance."""
        entry_r = min(self.entry_radii) if self.start is None else math.inf
        return _buoy_clearance(self.buoys, clearance, p, entry_r, min(self.exit_radii) if self.exit_radii else 9.0)


def _search(q: _Problem, p: RouteParams, level, need_red, need_green, banned=None):
    """One lattice search at one margin level. Returns the nodes [(x, y, yaw of the move that arrived)] or None.

    State: cell, heading (16), credits in hand (red, green). Moves: one lattice step in a heading at most
    max_turn_deg from the current one; the turn is made on the spot before the step and every heading it sweeps
    through must be allowed at that spot. The first node is on an entry circle with the heading the boat has as its
    loop closes; the last is on an exit circle, arriving within 90 deg of the counterclockwise direction.
    """
    clearance, shield_margin, credit_margin, near_range = level
    P, red, green, trusted, confirmed = q.P, q.red, q.green, q.trusted, q.confirmed
    s = p.grid_step
    lo = P.min(axis=0) - p.grid_margin
    hi = P.max(axis=0) + p.grid_margin
    if q.start is not None:                     # a start outside that water is still on the lattice
        lo = np.minimum(lo, q.start - p.start_snap)
        hi = np.maximum(hi, q.start + p.start_snap)
    nx, ny = int(math.ceil((hi[0] - lo[0]) / s)) + 1, int(math.ceil((hi[1] - lo[1]) / s)) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    C = np.stack([lo[0] + gx.ravel() * s, lo[1] + gy.ravel() * s], axis=1)       # cell centres, (N, 2)
    N, B = len(C), len(P)
    D = P[None] - C[:, None]                                                       # (N, B, 2) cell -> buoy
    dist = np.linalg.norm(D, axis=2)
    U = D / np.maximum(dist, 1e-9)[..., None]

    clear_b = q.clearances(clearance, p)
    free = np.all(dist >= clear_b[None], axis=1)
    if q.tree is not None:
        free &= q.tree.query(C)[0] >= p.obstacle_clearance
    if q.keep_out is not None:
        free &= ~q.keep_out(C)
    if banned is not None and len(banned):
        free &= cKDTree(np.asarray(banned, dtype=float)).query(C)[0] > 0.6 * s

    # With buoy b abeam (the side direction is then the direction to b): is it shielded, is it clearly the closest?
    proj = np.einsum("cbk,cjk->cbj", U, D)                  # [c, b, j]: how far buoy j is onto the side b is on
    dj = np.broadcast_to(dist[:, None, :], proj.shape)
    notself = ~np.eye(B, dtype=bool)[None]
    cand = notself & trusted[None, None, :] & (proj >= p.shield_lat_min)
    shielded = np.where(cand, dj, np.inf).min(axis=2) <= dist - shield_margin
    rival = notself & (proj > -p.credit_lat_slack)
    clear = (np.where(rival, dj, np.inf).min(axis=2) >= dist + credit_margin) & (dist <= p.credit_range) \
        & confirmed[None]
    danger = (~shielded) & (dist <= p.violation_range)      # being on the wrong side of the boat here would violate

    tol = math.sin(math.radians(p.heading_tolerance_deg + 4.0))
    cos_near = math.cos(math.radians(min(p.near_sector_deg + p.heading_tolerance_deg, 89.0)))
    close = dist <= near_range + 0.25
    ok = np.zeros((N, _K), dtype=bool)          # pose allowed
    fwd_ok = np.zeros((N, _K), dtype=bool)      # step from this pose allowed
    fwd_to = np.zeros((N, _K), dtype=np.int64)
    fwd_red = np.zeros((N, _K), dtype=bool)     # that step credits a red / a green
    fwd_green = np.zeros((N, _K), dtype=bool)
    fwd_len = np.zeros(_K)
    ci, cj = gx.ravel(), gy.ravel()
    for k, (di, dj_) in enumerate(_MOVES):
        u = np.array([math.cos(_YAW[k]), math.sin(_YAW[k])])
        along = D @ u
        lat = D @ np.array([-u[1], u[0]])
        port = lat > 0
        wrong = (port & red[None]) | (~port & green[None])
        right = (~port & red[None]) | (port & green[None])
        near_abeam = np.abs(along) <= tol * dist
        on_beam = close & (np.abs(along) <= cos_near * dist)
        ok[:, k] = free & ~np.any(wrong & ((near_abeam & danger) | on_beam), axis=1)
        ti, tj = ci + di, cj + dj_
        inside = (ti >= 0) & (ti < nx) & (tj >= 0) & (tj < ny)
        to = np.where(inside, ti * ny + tj, 0)
        fwd_to[:, k] = to
        fwd_len[k] = s * math.hypot(di, dj_)
        crossing = (along > 0) & (along - fwd_len[k] <= 0)                         # comes abeam during the step
        bad = np.any(crossing & wrong & (danger | danger[to] | close | close[to]), axis=1)
        good = crossing & right & clear & clear[to]
        fwd_red[:, k] = np.any(good & red[None], axis=1)
        fwd_green[:, k] = np.any(good & green[None], axis=1)
        mid_free = free
        if abs(di) + abs(dj_) > 2:              # a knight step passes between cells: check its midpoint too
            mid = C + 0.5 * fwd_len[k] * u
            mid_free = np.all(np.linalg.norm(P[None] - mid[:, None], axis=2) >= clear_b[None], axis=1)
        fwd_ok[:, k] = inside & ~bad & mid_free

    # graph: node = ((cell * K + k) * 4 + bits), bits = 1 red credited | 2 green credited
    dmin = dist.min(axis=1) if B else np.full(N, np.inf)
    soft = p.soft_weight * (np.clip(p.soft_clearance - dmin, 0.0, None)
                            / max(p.soft_clearance - clearance, 1e-6)) ** 2
    cells = np.arange(N)
    rows, cols, vals = [], [], []
    max_steps = max(0, int(round(p.max_turn_deg / 22.5)))
    for k in range(_K):                         # k: heading of the step
        to = fwd_to[:, k]
        move_ok = ok[:, k] & fwd_ok[:, k] & ok[to, k]
        move_cost = fwd_len[k] * (1.0 + 0.5 * (soft + soft[to]))
        gain = fwd_red[:, k].astype(int) | (fwd_green[:, k].astype(int) << 1)
        for sign in (1, -1):
            sweep = move_ok.copy()              # every heading swept through while turning on the spot must be ok
            for step in range(0, max_steps + 1):
                if step == 0 and sign < 0:
                    continue
                k0 = (k - sign * step) % _K     # heading before the turn
                sweep = sweep & ok[:, k0]
                c0 = cells[sweep]
                if not len(c0):
                    break
                turn = abs(float(wrap(_YAW[k] - _YAW[k0])))
                cost = move_cost[c0] + (p.turn_cost * turn + p.corner_cost if step else 0.0)
                for bits in range(4):
                    rows.append((c0 * _K + k0) * 4 + bits)
                    cols.append((to[c0] * _K + k) * 4 + (bits | gain[c0]))
                    vals.append(cost)
    n_nodes = N * _K * 4 + 1
    source = n_nodes - 1

    if q.start is not None:
        # start: the allowed cells near the given position, the nearest strongly preferred, facing any way (the boat
        # turns there before it is judged again)
        d0 = np.linalg.norm(C - q.start[None], axis=1)
        near = d0 <= p.start_snap
        any_start = False
        for k in range(_K):
            sc = cells[near & ok[:, k]]
            if not len(sc):
                continue
            any_start = True
            rows.append(np.full(len(sc), source))
            cols.append((sc * _K + k) * 4)
            vals.append(1e-6 + 10.0 * d0[sc])
        if not any_start:
            return None
    else:
        # start: where an entry loop closes, with the heading the boat arrives there with (clockwise, last chord)
        e = P[q.entry]
        rel = C - e
        r = np.linalg.norm(rel, axis=1)
        phi = np.arctan2(rel[:, 1], rel[:, 0])
        on_ring = np.zeros(N, dtype=bool)
        for R in q.entry_radii:
            on_ring |= np.abs(r - R) <= p.ring_tolerance
        k_arr = np.abs(wrap(_YAW[None, :] - arrival_yaw(phi, 1, p.circle_points)[:, None])).argmin(axis=1)
        sc = cells[on_ring & ok[cells, k_arr]]
        if not len(sc):
            return None
        rows.append(np.full(len(sc), source))
        cols.append((sc * _K + k_arr[sc]) * 4)
        vals.append(np.full(len(sc), 1e-6))
    G = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_nodes, n_nodes))
    dist_to, pred = dijkstra(G, directed=True, indices=source, return_predecessors=True)

    want = (1 if need_red else 0) | (2 if need_green else 0)
    bit_sets = [b for b in range(4) if b & want == want]
    if q.exit_ is not None:
        relx = C - P[q.exit_]
        rx = np.linalg.norm(relx, axis=1)
        phix = np.arctan2(relx[:, 1], relx[:, 0])
        ring = np.zeros(N, dtype=bool)
        for R in q.exit_radii:
            ring |= np.abs(rx - R) <= p.ring_tolerance
        gc = cells[ring]
        goals = np.concatenate([((gc[:, None] * _K + np.arange(_K)[None]) * 4 + b).ravel() for b in bit_sets])
        if len(goals):      # arriving across the circle's direction costs the turn; much more than max_turn_deg is not on
            off = np.abs(wrap(_YAW[(goals // 4) % _K] - (phix[(goals // 4) // _K] + math.pi / 2.0)))
            dist_to = dist_to.copy()
            dist_to[goals] += np.where(off <= math.radians(p.max_turn_deg) + 0.25, p.turn_cost * off, np.inf)
    else:                   # no exit known: stop once the credits are in hand
        allc = cells[free]
        goals = np.concatenate([((allc[:, None] * _K + np.arange(_K)[None]) * 4 + b).ravel() for b in bit_sets])
    if not len(goals):
        return None
    best = goals[np.argmin(dist_to[goals])]
    if not np.isfinite(dist_to[best]):
        return None
    chain = []
    node = int(best)
    while node != source and node >= 0:
        chain.append(node)
        node = int(pred[node])
    chain.reverse()
    return [(C[(n // 4) // _K, 0], C[(n // 4) // _K, 1], float(_YAW[(n // 4) % _K])) for n in chain]


def _corners(nodes) -> list:
    """Lattice nodes -> the points where the boat stops and turns (the start and the end included)."""
    pts = [np.array(nodes[0][:2])]
    for i in range(1, len(nodes) - 1):
        if abs(float(wrap(nodes[i + 1][2] - nodes[i][2]))) > 1e-6:
            pts.append(np.array(nodes[i][:2]))
    if len(nodes) > 1:
        pts.append(np.array(nodes[-1][:2]))
    return pts


def _doubling_back(points, start_yaw, p: RouteParams) -> list:
    """Corners where the route turns too sharply, or turns the same way by more than double_back_deg within
    double_back_window metres of path (a small loop to collect or dodge a pass). Returns those corner points."""
    pts = np.asarray(points, dtype=float)[:, :2]
    legs = np.diff(pts, axis=0)
    L = np.linalg.norm(legs, axis=1)
    turns = wrap(np.diff(np.concatenate([[start_yaw], np.arctan2(legs[:, 1], legs[:, 0])])))   # turn at corner i
    at = np.concatenate([[0.0], np.cumsum(L)])[:-1]                                              # path position
    bad = [tuple(pts[i]) for i in np.nonzero(np.abs(turns) > math.radians(p.max_turn_deg) + 0.3)[0]]
    for i in range(len(turns)):
        total = 0.0
        for j in range(i, len(turns)):
            if at[j] - at[i] > p.double_back_window:
                break
            total += turns[j]
            if abs(total) > math.radians(p.double_back_deg):
                bad += [tuple(pts[k]) for k in range(i, j + 1)]
                break
    return bad


def _acceptable(points, start_yaw, q: _Problem, clear_b, p: RouteParams, strict: RouteParams, need_red, need_green):
    """Fine check of a waypoint polyline: turn limit, clearance, and the strict rule at the nominal heading and at
    +- heading_tolerance. Returns (dense poses, strict CheckResult) or None."""
    dense = densify(points, start_yaw, p)
    if not len(dense):
        return None
    if _doubling_back(points, start_yaw, p):
        return None
    d = np.linalg.norm(dense[:, None, :2] - q.P[None], axis=2)
    if np.any(d.min(axis=0) < clear_b - 1e-6):
        return None
    if q.tree is not None and q.tree.query(dense[:, :2])[0].min() < p.obstacle_clearance - 1e-6:
        return None
    nominal = None
    for off in (0.0, math.radians(p.heading_tolerance_deg), -math.radians(p.heading_tolerance_deg)):
        res = check_trajectory(dense, q.buoys, strict, yaw_offset=off)
        if not res.satisfied(need_red, need_green):
            return None
        if off == 0.0:
            nominal = res
    return dense, nominal


def _plan_level(q: _Problem, p: RouteParams, li: int, need_red: bool, need_green: bool) -> Optional[RoutePlan]:
    """Lattice search at one margin level, fine check (rerunning the search around what fails it), corner
    shortcuts, leg splitting."""
    level = p.levels[li]
    strict = replace(p, clearance=level[0], shield_margin=level[1], credit_margin=level[2], near_range=level[3])
    clear_b = q.clearances(level[0], p) - 0.1
    from_start = q.start is not None
    e = None if from_start else q.P[q.entry]
    n = p.circle_points

    def start_yaw_of(points):
        """Heading as the judged path begins: arriving along the entry loop's last chord, or, from a given start,
        already facing the first leg."""
        first = points[0]
        if from_start:
            return math.atan2(points[1][1] - first[1], points[1][0] - first[0])
        return arrival_yaw(math.atan2(first[1] - e[1], first[0] - e[0]), 1, n)

    def acceptable(points):
        return _acceptable(points, start_yaw_of(points), q, clear_b, p, strict, need_red, need_green)

    banned, got, pts = [], None, None
    for attempt in range(p.search_retries + 1):
        nodes = _search(q, p, level, need_red, need_green, banned)
        if nodes is not None and len(nodes) == 1 and from_start and q.exit_ is not None:
            return _already_joined(q, p, li, nodes[0], need_red, need_green)
        if nodes is None or len(nodes) < 2:
            if p.debug:
                print(f"   level {li} red {need_red} green {need_green}: no lattice route (attempt {attempt})")
            return None
        pts = _corners(nodes)
        got = acceptable(pts)
        if q.exit_ is None:
            # the route stops as the last buoy comes abeam: run on so the pass is beyond doubt
            u = (pts[-1] - pts[-2]) / max(float(np.linalg.norm(pts[-1] - pts[-2])), 1e-9)
            for extra in (1.0, 1.5, 0.5):
                trial = pts[:-1] + [pts[-1] + u * extra]
                g = acceptable(trial)
                if g is not None:
                    pts, got = trial, g
                    break
        if got is not None:
            break
        # keep the next search away from where this route went wrong
        dense = densify(pts, start_yaw_of(pts), p)
        where = []
        for off in (0.0, 1.0, -1.0):
            r = check_trajectory(dense, q.buoys, strict, yaw_offset=off * math.radians(p.heading_tolerance_deg))
            where += [v.pose[:2] for v in r.violations]
            if p.debug:
                print(f"   level {li} attempt {attempt} offset {off:+.0f}: {r.summary()} | "
                      + "; ".join(str(v) for v in r.violations))
        d = np.linalg.norm(dense[:, None, :2] - q.P[None], axis=2)
        where += [tuple(dense[i, :2]) for i in np.unique(np.nonzero(d < clear_b[None])[0])]
        where += _doubling_back(pts, start_yaw_of(pts), p)
        if not where:       # the credits did not register, or a corner is too sharp: nothing to steer away from
            return None
        banned += where
    if got is None:
        return None
    i = 0
    while i < len(pts) - 2:                                 # shortcut corners while the fine check still holds
        j = len(pts) - 1
        while j > i + 1:
            trial = pts[:i + 1] + pts[j:]
            g = acceptable(trial)
            if g is not None:
                pts, got = trial, g
                break
            j -= 1
        i += 1
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
        status="ok", waypoints=waypoints, start_yaw=float(wrap(start_yaw_of(out))), poses=dense, check=res,
        passes=[(ev.name or str(ev.index), ev.state, ev.side, round(ev.distance, 2), ev.verdict)
                for ev in res.events],
        red_passed=sorted({ev.name or str(ev.index) for ev in res.red_credits}),
        green_passed=sorted({ev.name or str(ev.index) for ev in res.green_credits}),
        level=li, clearance=clearance, margin=float(min(margins)), length=path_length(waypoints),
        entry_radius=entry_r, entry_start_angle=start_angle, need_red=need_red, need_green=need_green)
    if not from_start:
        # the entry loop: begins and closes at waypoints[0]; its last goal already faces the first transit leg
        loop = circle_goals(e, entry_r, start_angle, 1, n, 0)
        loop[0] = (waypoints[0][0], waypoints[0][1], loop[0][2])
        loop[-1] = waypoints[0]
        plan.entry_circle = loop
    if q.exit_ is not None:
        x = q.P[q.exit_]
        plan.exit_radius = min(q.exit_radii, key=lambda R: abs(R - float(np.linalg.norm(last - x))))
        plan.exit_join_angle = math.atan2(last[1] - x[1], last[0] - x[0])
        ring = circle_goals(x, plan.exit_radius, plan.exit_join_angle, -1, n, p.exit_overshoot_points)
        plan.exit_circle = ring[1:]                         # its point 0 is the last transit waypoint
    _set_goals(plan)
    return plan


def _already_joined(q: _Problem, p: RouteParams, li: int, node, need_red: bool, need_green: bool) -> RoutePlan:
    """A start that is on the exit circle with nothing left to pass: the transit is that one point."""
    x = q.P[q.exit_]
    at = np.array(node[:2], dtype=float)
    radius = min(q.exit_radii, key=lambda R: abs(R - float(np.linalg.norm(at - x))))
    angle = math.atan2(at[1] - x[1], at[0] - x[0])
    ring = circle_goals(x, radius, angle, -1, p.circle_points, p.exit_overshoot_points)
    here = (float(at[0]), float(at[1]), ring[0][2])
    poses = np.array([here], dtype=float)
    strict = replace(p, clearance=p.levels[li][0], shield_margin=p.levels[li][1], credit_margin=p.levels[li][2],
                     near_range=p.levels[li][3])
    plan = RoutePlan(status="ok", waypoints=[here], start_yaw=here[2], poses=poses,
                     check=check_trajectory(poses, q.buoys, strict), level=li, clearance=math.inf, margin=math.inf,
                     need_red=need_red, need_green=need_green, exit_radius=radius, exit_join_angle=angle,
                     exit_circle=ring[1:])
    _set_goals(plan)
    return plan


def _set_goals(plan: RoutePlan):
    """plan.goals from its three parts. The entry loop ends on waypoints[0]; without one, waypoints[0] is a goal."""
    transit = plan.waypoints[1:] if plan.entry_circle else plan.waypoints
    plan.goals = ([(g[0], g[1], g[2], "entry") for g in plan.entry_circle]
                  + [(g[0], g[1], g[2], "transit") for g in transit]
                  + [(g[0], g[1], g[2], "exit") for g in plan.exit_circle])


def _merge_legs(plan: RoutePlan):
    """One goal per corner: drop the waypoints that only split a straight leg."""
    w = plan.waypoints
    keep = [w[0]] + [w[i] for i in range(1, len(w) - 1) if abs(float(wrap(w[i][2] - w[i - 1][2]))) > 1e-6] + [w[-1]]
    plan.waypoints = keep if len(w) > 1 else w
    _set_goals(plan)


def judged_poses(plan: RoutePlan, buoys: Sequence[Buoy], p: Optional[RouteParams] = None,
                 entry_immune: bool = True, exit_immune: bool = True) -> np.ndarray:
    """The planned poses that are judged: from the point the entry loop closes to the point the boat joins the exit
    circle; with a circle not immune, its lap (as the polygon the node drives) is included."""
    p = p or RouteParams()
    parts = []
    if not entry_immune and plan.entry_circle:
        lap = densify(plan.entry_circle, None, p)
        parts.append(lap)
    parts.append(plan.poses)
    if not exit_immune and plan.exit_circle:
        ring = densify([plan.waypoints[-1]] + plan.exit_circle, float(plan.poses[-1, 2]), p)
        parts.append(ring)
    return np.vstack(parts)


def check_plan(plan: RoutePlan, buoys: Sequence[Buoy], p: Optional[RouteParams] = None,
               entry_immune: bool = True, exit_immune: bool = True) -> CheckResult:
    """The literal rule over the judged part of a plan (see judged_poses)."""
    p = p or RouteParams()
    return check_trajectory(judged_poses(plan, buoys, p, entry_immune, exit_immune), buoys, p.judge())


def plan_route(buoys: Sequence[Buoy], params: Optional[RouteParams] = None, obstacles=None, keep_out=None,
               entry_immune: bool = True, exit_immune: bool = True, start=None, need_red: bool = True,
               need_green: bool = True) -> RoutePlan:
    """Plan entry circle -> transit -> exit circle.

    start (x, y): resume a run whose entry loop is done. No entry loop is planned; the transit begins on the nearest
    allowed lattice cell within start_snap of it (waypoints[0], a goal like the others), facing any way.
    need_red / need_green False: that pass is already in hand and is not looked for again.

    What is judged (defaults): the path from the point where the entry loop closes (the loop begins there, goes
    round clockwise and returns there; the loop and the approach before it are immune) to the point where the boat
    joins the exit circle (immune from then on). The turn the boat makes at the closing point is judged.
    entry_immune / exit_immune False: that circle's lap is judged too.

    buoys      every buoy and buoy-like object: ENTRY, EXIT, RED, GREEN, BLACK, UNKNOWN (Buoy)
    obstacles  (M, 2) points to keep obstacle_clearance from (lethal costmap cells that are not buoys), or None
    keep_out   optional f((N, 2) points) -> (N,) bool, water the transit may not use

    Returns a RoutePlan. status:
      ok                both a red to starboard and a green to port on the transit, no violation
      ok_partial        no violation, but only what could be had (message says what is missing and why)
      missing_exit      no exit buoy: the route passes one red and one green correctly and stops
      missing_entry     no entry buoy: nothing planned
      no_route          no compliant transit at any margin level
      circle_violation  a circle that is not immune passes a buoy on the wrong side whatever is done; the plan is
                        otherwise as for ok, the message names the buoys
    """
    p = params or RouteParams()
    buoys = list(buoys)
    P, red, green, trusted, confirmed = _arrays(buoys)
    entry = next((i for i, b in enumerate(buoys) if b.state == ENTRY), None)
    exit_ = next((i for i, b in enumerate(buoys) if b.state == EXIT), None)
    from_start = start is not None
    if from_start:
        entry_radii = []
    elif entry is None:
        return RoutePlan("missing_entry", "no entry (flashing blue) buoy")
    else:
        entry_radii = [p.entry_radius] if p.entry_radius else radius_candidates(buoys, entry, p)
    exit_radii = ([p.exit_radius] if p.exit_radius else radius_candidates(buoys, exit_, p)) if exit_ is not None \
        else []
    circle_notes = []

    def fewest(index, radii, sigma, what):
        """A circle that is judged: keep the radii whose lap passes the fewest buoys on the wrong side."""
        laps = {R: circle_alone(buoys, index, R, sigma, p) for R in radii}
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
    q = _Problem(buoys, P, red, green, trusted, confirmed, entry, exit_, entry_radii, exit_radii, obstacles, tree,
                 keep_out, np.asarray(start, dtype=float)[:2] if from_start else None)
    owed_red, owed_green = need_red, need_green     # what the run still has to earn
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
    asks = [(want_red, want_green)]     # most wanted first: both colours, then one, then just a clean transit
    if want_red and want_green:
        asks += [(True, False), (False, True)]
    if want_red or want_green:
        asks.append((False, False))
    fragile = []
    for need_red, need_green in asks:
        candidates = []
        for li in range(len(p.levels)):
            plan = _plan_level(q, p, li, need_red, need_green)
            if plan is None:
                continue
            if any(np.allclose(np.asarray(plan.waypoints), np.asarray(c.waypoints)) for c in candidates
                   if len(c.waypoints) == len(plan.waypoints)):
                continue
            candidates.append(plan)
            if p.choose != "robust":
                break
        if not candidates:
            continue
        for c in candidates:
            c.judge = check_plan(c, buoys, p, entry_immune, exit_immune)
            near = [e for e in c.judge.events if e.closest and e.state in (RED, GREEN) and min(
                np.hypot(e.pose[0] - c.waypoints[0][0], e.pose[1] - c.waypoints[0][1]),
                np.hypot(e.pose[0] - c.waypoints[-1][0], e.pose[1] - c.waypoints[-1][1])) <= p.handover_distance]
            c.handover = [str(e) for e in near]
        if p.mc_trials > 0:
            for c in candidates:
                c.mc = monte_carlo(c, buoys, p.mc_buoy_sigma, p.mc_trials, seed=1, params=p)
            fragile += [c for c in candidates if c.mc["full"] < p.min_mc_full]
            candidates = [c for c in candidates if c.mc["full"] >= p.min_mc_full]
            if not candidates:
                continue
            # most often fully compliant; within 2 % of the best: the quieter hand-overs, then the stricter level
            best = max(c.mc["full"] - 0.2 * c.mc["strike"] for c in candidates)
            candidates = [c for c in candidates if c.mc["full"] - 0.2 * c.mc["strike"] >= best - 0.02]
            candidates.sort(key=lambda c: (len(c.handover), c.level))
        plan = candidates[0]
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
            _merge_legs(plan)
        return plan
    why = "no compliant transit at any margin level"
    if fragile:
        best = max(fragile, key=lambda c: c.mc["full"])
        why = (f"the only transits found are fragile: the best fully complies in {best.mc['full'] * 100:.0f}% of "
               f"simulated executions (< {p.min_mc_full * 100:.0f}%)")
    return RoutePlan("no_route", "; ".join(notes + circle_notes + [why]), circle_violations=circle_notes)


# ----------------------------------------------------------------------------------------------------------------
# Execution model and Monte Carlo
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ExecutionNoise:
    """How ASV5 executes a list of TrackPath goals, from trackpath_17-21-27 and trackpath_10-57-07."""
    arrival_sigma: float = 0.25       # m per axis: where it stops relative to the goal (measured median 0.14-0.26 m,
    #                                   90% 0.43-0.49, max 0.99 with a 1.0 m tolerance)
    bulge_sigma: float = 0.3          # m: largest sideways departure from the straight leg (legs <= 3.5 m: median
    #                                   0.11-0.16, 90% 0.45, max 1.1)
    crab_sigma_deg: float = 6.0       # deg: heading minus direction of travel while moving (sd 3.8-6.9)
    crab_corr: float = 0.9            # sample to sample correlation of that at sample_step
    yaw_stop_sigma_deg: float = 6.0   # deg: heading where it stops relative to the commanded one (90% 7.8)
    overshoot_turn_deg: float = 5.0   # deg: turns on the spot overshoot by up to about this


def simulate_execution(waypoints, start_yaw: Optional[float], rng, noise: Optional[ExecutionNoise] = None,
                       params: Optional[RouteParams] = None, final_turn: bool = False) -> np.ndarray:
    """One noisy execution of the waypoints as stop-and-go TrackPath goals -> poses (N, 3)."""
    noise = noise or ExecutionNoise()
    p = params or RouteParams()
    W = np.asarray(waypoints, dtype=float)
    at = W[0, :2] + rng.normal(0.0, noise.arrival_sigma, 2)
    if start_yaw is None:   # the transit starts once the boat has turned away from the circle
        first = W[1, :2] - at
        start_yaw = math.atan2(first[1], first[0])
    yaw = start_yaw + math.radians(rng.normal(0.0, noise.yaw_stop_sigma_deg))
    out = [(at[0], at[1], yaw)]
    dth = math.radians(p.turn_step_deg)

    def turn(to):
        nonlocal yaw
        delta = float(wrap(to - yaw))
        over = math.radians(abs(rng.normal(0.0, noise.overshoot_turn_deg))) * (1 if delta >= 0 else -1)
        for target in (yaw + delta + over, yaw + delta):
            d2 = target - yaw
            n = max(1, int(math.ceil(abs(d2) / dth)))
            for i in range(1, n + 1):
                out.append((at[0], at[1], yaw + d2 * i / n))
            yaw = target

    for i in range(1, len(W)):
        target = W[i, :2] + rng.normal(0.0, noise.arrival_sigma, 2)
        leg = target - at
        L = float(np.linalg.norm(leg))
        if L < 1e-3:
            continue
        direction = math.atan2(leg[1], leg[0])
        turn(direction + math.radians(rng.normal(0.0, noise.crab_sigma_deg)))
        n = max(2, int(math.ceil(L / p.sample_step)))
        u = leg / L
        nrm = np.array([-u[1], u[0]])
        amp = rng.normal(0.0, noise.bulge_sigma)
        crab = wrap(yaw - direction)
        s = np.arange(1, n + 1) / n
        lat = amp * np.sin(math.pi * s)
        pts = at[None] + np.outer(s * L, u) + np.outer(lat, nrm)
        travel = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0])) if n > 2 else np.full(n, direction)
        sd = math.radians(noise.crab_sigma_deg)
        for j in range(n):
            crab = noise.crab_corr * crab + math.sqrt(1 - noise.crab_corr ** 2) * rng.normal(0.0, sd)
            yaw = travel[j] + crab
            out.append((pts[j, 0], pts[j, 1], yaw))
        at = target
        if final_turn and len(W[i]) > 2 and i == len(W) - 1:
            turn(W[i, 2] + math.radians(rng.normal(0.0, noise.yaw_stop_sigma_deg)))
    return np.array(out, dtype=float)


def monte_carlo(plan: RoutePlan, buoys: Sequence[Buoy], buoy_sigma: float, trials: int = 500, seed: int = 0,
                noise: Optional[ExecutionNoise] = None, params: Optional[RouteParams] = None,
                unknown_exist: Optional[bool] = None, common_sigma: float = 0.0) -> dict:
    """Execute the plan `trials` times against buoys moved by N(0, buoy_sigma) per axis (each on its own, plus a
    shared shift of common_sigma) and judge each run by the literal rule.

    unknown_exist: True = UNKNOWN objects are real black buoys, False = they are not there, None = a coin per trial.
    Returns rates: compliant (no violation), red, green, full (no violation and both credits), strike.
    """
    p = params or RouteParams()
    judge = p.judge()
    rng = np.random.default_rng(seed)
    n = dict(compliant=0, red=0, green=0, full=0, strike=0)
    worst = []
    for _ in range(trials):
        shift = rng.normal(0.0, common_sigma, 2) if common_sigma > 0 else np.zeros(2)
        truth = []
        for b in buoys:
            if b.state == UNKNOWN:
                exists = unknown_exist if unknown_exist is not None else rng.random() < 0.5
                if not exists:
                    continue
            d = rng.normal(0.0, buoy_sigma, 2) + shift
            truth.append(replace(b, x=b.x + d[0], y=b.y + d[1], state=BLACK if b.state == UNKNOWN else b.state))
        poses = simulate_execution(plan.waypoints, plan.start_yaw, rng, noise, p)
        res = check_trajectory(poses, truth, judge)
        ok = res.compliant
        r, g = bool(res.red_credits) or not plan.need_red, bool(res.green_credits) or not plan.need_green
        n["compliant"] += ok
        n["red"] += r
        n["green"] += g
        n["full"] += ok and r and g
        n["strike"] += bool(res.strikes)
        if not ok:
            worst += [(ev.name or str(ev.index), ev.side) for ev in res.violations]
    out = {k: v / trials for k, v in n.items()}
    out["violators"] = {k: worst.count(k) for k in set(worst)}
    return out
