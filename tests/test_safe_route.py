"""safe_route: the Safe Passage planner and pass rule. A channel running north: reds to the east, greens to the west."""

import math

import numpy as np
import pytest

from frames.utils import safe_route as sr
from frames.utils.safe_route import BLACK, ENTRY, EXIT, GREEN, RED, Buoy, RouteParams

FAST = dict(mc_trials=0, choose="strictest")  # one search per plan: the selection between levels is not under test


def channel(entry=True, exit_=True):
    buoys = [Buoy(3.0, 7.0, RED, name="r1"), Buoy(-3.0, 7.0, GREEN, name="g1"),
             Buoy(3.0, 13.0, RED, name="r2"), Buoy(-3.0, 13.0, GREEN, name="g2"),
             Buoy(9.0, 10.0, BLACK, name="k")]
    if entry:
        buoys.append(Buoy(0.0, 0.0, ENTRY, name="entry"))
    if exit_:
        buoys.append(Buoy(0.0, 20.0, EXIT, name="exit"))
    return buoys


def literal(waypoints, buoys, start_yaw=None):
    return sr.check_trajectory(sr.densify(waypoints, start_yaw), buoys, RouteParams().judge())


@pytest.fixture(scope="module")
def full():
    return sr.plan_route(channel(), RouteParams(**FAST))


def test_full_plan_circles_and_passes(full):
    assert full.status == "ok"
    assert full.entry_circle[0][:2] == pytest.approx(full.waypoints[0][:2])   # the loop begins and closes there
    assert full.entry_circle[-1] == full.waypoints[0]
    assert len(full.entry_circle) == RouteParams().circle_points + 1
    assert full.exit_circle
    assert [g[3] for g in full.goals] == (["entry"] * len(full.entry_circle) + ["transit"] * (len(full.waypoints) - 1)
                                          + ["exit"] * len(full.exit_circle))
    res = literal(full.waypoints, channel(), full.start_yaw)
    assert res.compliant and res.red_credits and res.green_credits


def test_one_goal_per_corner(full):
    split = sr.plan_route(channel(), RouteParams(merge_legs=False, **FAST))
    assert len(full.waypoints) < len(split.waypoints)
    assert sr.path_length(full.waypoints) == pytest.approx(sr.path_length(split.waypoints))
    yaws = [w[2] for w in full.waypoints[:-1]]
    assert all(abs(float(sr.wrap(b - a))) > 1e-6 for a, b in zip(yaws[:-2], yaws[1:-1]))   # every goal is a corner
    assert split.entry_circle == full.entry_circle and split.exit_circle == full.exit_circle


def test_resume_from_a_start_has_no_entry_loop():
    start = (0.4, 3.2)
    plan = sr.plan_route(channel(), RouteParams(**FAST), start=start)
    assert plan.status == "ok" and not plan.entry_circle and plan.exit_circle
    assert plan.goals[0][3] == "transit" and plan.goals[0][:3] == plan.waypoints[0]
    assert math.dist(plan.waypoints[0][:2], start) <= RouteParams().start_snap
    res = literal(plan.waypoints, channel())
    assert res.compliant and res.red_credits and res.green_credits


def test_resume_does_not_look_for_passes_in_hand():
    buoys = channel()
    start = (0.0, 16.0)   # past both pairs: nothing is left to pass on the way to the exit
    owed = sr.plan_route(buoys, RouteParams(**FAST), start=start)
    done = sr.plan_route(buoys, RouteParams(**FAST), start=start, need_red=False, need_green=False)
    assert done.status == "ok" and not done.need_red and not done.need_green
    assert literal(done.waypoints, buoys).compliant
    assert owed.status != "ok" or sr.path_length(owed.waypoints) > sr.path_length(done.waypoints) + 5.0


def test_start_outside_the_searched_water_is_reached():
    plan = sr.plan_route(channel(), RouteParams(**FAST), start=(0.0, -25.0))
    assert plan.ok and math.dist(plan.waypoints[0][:2], (0.0, -25.0)) <= RouteParams().start_snap
    assert literal(plan.waypoints, channel()).compliant


def test_no_entry_needs_a_start():
    buoys = channel(entry=False)
    assert sr.plan_route(buoys, RouteParams(**FAST)).status == "missing_entry"
    plan = sr.plan_route(buoys, RouteParams(**FAST), start=(0.0, 0.0))
    assert plan.status == "ok" and not plan.entry_circle and plan.exit_circle


def test_no_exit():
    buoys = channel(exit_=False)
    plan = sr.plan_route(buoys, RouteParams(**FAST))
    assert plan.status == "missing_exit" and plan.entry_circle and not plan.exit_circle
    res = literal(plan.waypoints, buoys, plan.start_yaw)
    assert res.compliant and res.red_credits and res.green_credits
    nothing = sr.plan_route(buoys, RouteParams(**FAST), start=(0.0, 16.0), need_red=False, need_green=False)
    assert nothing.status == "missing_exit" and not nothing.goals


def test_pass_tracker_follows_the_live_pose():
    tracker = sr.PassTracker(channel(), RouteParams())
    events = []
    for y in np.arange(2.0, 18.0, 0.1):
        events += tracker.update((0.0, y, math.pi / 2.0))
    res = tracker.result()
    assert {e.name for e in res.red_credits} == {"r1", "r2"} and {e.name for e in res.green_credits} == {"g1", "g2"}
    assert not res.violations
    wrong = sr.PassTracker(channel(), RouteParams())
    for y in np.arange(18.0, 2.0, -0.1):   # the same water southbound: every pass is on the wrong side
        wrong.update((0.0, y, -math.pi / 2.0))
    assert wrong.result().violations and not wrong.result().red_credits


def test_resume_on_the_exit_circle_goes_round_it():
    buoys = channel()
    plan = sr.plan_route(buoys, RouteParams(**FAST), start=(3.0, 20.0), need_red=False, need_green=False)
    assert plan.status == "ok" and len(plan.waypoints) == 1 and plan.exit_circle
    assert [g[3] for g in plan.goals] == ["transit"] + ["exit"] * len(plan.exit_circle)
    assert math.dist(plan.waypoints[0][:2], (3.0, 20.0)) < 0.5
    full = sr.plan_route(buoys, RouteParams(), start=(3.0, 20.0), need_red=False, need_green=False)   # Monte Carlo on
    assert full.status == "ok" and full.exit_circle
