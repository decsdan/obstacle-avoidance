"""Unit tests for the per-episode terminal classifier."""

import pytest

from scenario_runner.terminal_detector import TerminalConfig, TerminalDetector


def _make(goal_tolerance=0.2, timeout=10.0, stuck_window=2.0, stuck_dist=0.05):
    cfg = TerminalConfig(
        goal_tolerance=goal_tolerance,
        timeout_sec=timeout,
        stuck_window_sec=stuck_window,
        stuck_distance_m=stuck_dist,
    )
    return TerminalDetector(cfg, tick_hz=10.0)


def test_no_terminal_while_moving_toward_goal():
    d = _make()
    assert d.update(0.0, (0.0, 0.0), (5.0, 0.0)) is None
    assert d.update(0.1, (0.5, 0.0), (5.0, 0.0)) is None


def test_reached_when_within_tolerance():
    d = _make(goal_tolerance=0.2)
    assert d.update(0.0, (0.0, 0.0), (5.0, 0.0)) is None
    assert d.update(0.1, (4.95, 0.05), (5.0, 0.0)) == 'reached'


def test_timeout_when_sim_time_elapses():
    d = _make(timeout=1.0)
    assert d.update(0.0, (0.0, 0.0), (10.0, 0.0)) is None
    assert d.update(1.5, (0.5, 0.0), (10.0, 0.0)) == 'timeout'


def test_collision_is_latched_and_classified_first():
    d = _make()
    d.note_collision()
    # Even at the goal, collision wins.
    assert d.update(0.0, (5.0, 0.0), (5.0, 0.0)) == 'collision'


def test_terminal_is_sticky():
    d = _make(goal_tolerance=0.2)
    assert d.update(0.0, (5.0, 0.0), (5.0, 0.0)) == 'reached'
    assert d.update(1.0, (0.0, 0.0), (5.0, 0.0)) == 'reached'


def test_stuck_when_window_progress_below_threshold():
    d = _make(stuck_window=0.5, stuck_dist=0.1)
    # Ten ticks at 10 Hz fills the window; pose barely moves.
    for i in range(10):
        outcome = d.update(i * 0.1, (i * 0.001, 0.0), (10.0, 0.0))
    # Eleventh tick triggers stuck.
    assert d.update(1.1, (0.011, 0.0), (10.0, 0.0)) == 'stuck'


def test_cancel_classifies_cancelled():
    d = _make()
    d.note_cancel()
    assert d.update(0.0, (0.0, 0.0), (5.0, 0.0)) == 'cancelled'
