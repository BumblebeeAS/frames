"""Offline checks for the buoy gate manager's offset estimator, dedup and FFT classifier."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np

spec = importlib.util.spec_from_file_location(
    'buoy_gate_manager', Path(__file__).parents[1] / 'scripts/buoy_gate_manager_node.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

Color = module.Color
DetectedBuoyColor = module.DetectedBuoyColor
RATE = 10.0


def make_manager():
    # Mirrors cfg/asv5_buoy_gate_manager.yaml
    return module.TrackManager(
        target_freq=0.5, target_freq_tol=0.1, fft_min_magnitude=0.15,
        butterworth_cutoff_freq=3.0, virtual_sampling_rate=RATE, logger=Mock(),
    )


def add_track(manager, colors, t0=0.0, pos=(0.0, 0.0, 0.0)):
    track = module.BuoyTrack()
    manager.tracks[track.track_id] = track
    for i, color in enumerate(colors):
        manager._add_observation_to_track(
            track.track_id, color, np.array(pos), t0 + i / RATE
        )
    return track


def flasher(freq, color, seconds=10.0):
    t = np.arange(0.0, seconds, 1.0 / RATE)
    return [color if (ti * freq) % 1.0 < 0.5 else Color.BLACK for ti in t]


def test_offset_estimator_median_gate_and_freeze():
    est = module.OffsetEstimator(required_samples=5, outlier_gate=1.0)
    assert est.estimate is None
    for s in ([2.0, 1.0, 0.0], [2.2, 1.1, 0.0], [1.8, 0.9, 0.0]):
        assert est.add(np.array(s))
    assert not est.add(np.array([10.0, 10.0, 0.0]))  # Outlier
    assert est.add(np.array([2.1, 1.0, 0.0]))
    assert est.add(np.array([1.9, 1.0, 0.0]))
    assert est.is_sufficient
    assert not est.add(np.array([2.0, 1.0, 0.0]))  # Frozen
    np.testing.assert_allclose(est.estimate, [2.0, 1.0, 0.0])


def test_dedup_keeps_most_recently_observed():
    manager = make_manager()
    old = add_track(manager, [Color.BLUE] * 5, t0=0.0)
    recent = add_track(manager, [Color.BLUE] * 5, t0=100.0)
    old.published_state = recent.published_state = DetectedBuoyColor.SOLID_BLUE
    manager.deduplicate_singletons()
    assert list(manager.tracks) == [recent.track_id]
    assert len(recent.pose_history) == 10


def test_classifies_half_hz_red_flasher():
    manager = make_manager()
    track = add_track(manager, flasher(0.5, Color.RED))
    manager.classify_tracks()
    assert track.published_state == DetectedBuoyColor.FLASHING_RED


def test_constant_blue_is_solid_blue():
    manager = make_manager()
    track = add_track(manager, [Color.BLUE] * 100)
    manager.classify_tracks()
    assert track.published_state == DetectedBuoyColor.SOLID_BLUE


def test_two_hz_flasher_is_not_flashing():
    manager = make_manager()
    track = add_track(manager, flasher(2.0, Color.RED))
    manager.classify_tracks()
    assert track.published_state not in (
        DetectedBuoyColor.FLASHING_RED,
        DetectedBuoyColor.FLASHING_GREEN,
        DetectedBuoyColor.FLASHING_BLUE,
    )
