"""Unit tests for the log-odds inverse sensor model."""

import math

import numpy as np
import pytest

from obstacle_grid import log_odds


def test_decay_pulls_toward_zero():
    grid = np.array([[5.0, -3.0]], dtype=np.float32)
    log_odds.decay(grid, dt=1.0, tau=1.0)
    assert grid[0, 0] == pytest.approx(5.0 * math.exp(-1.0), rel=1e-5)
    assert grid[0, 1] == pytest.approx(-3.0 * math.exp(-1.0), rel=1e-5)


def test_decay_noop_on_zero_or_negative_dt():
    grid = np.array([[5.0]], dtype=np.float32)
    log_odds.decay(grid, dt=0.0, tau=1.0)
    log_odds.decay(grid, dt=-1.0, tau=1.0)
    log_odds.decay(grid, dt=1.0, tau=0.0)
    assert grid[0, 0] == pytest.approx(5.0)


def test_to_probability_sigmoid_round_trip():
    grid = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
    p = log_odds.to_probability(grid)
    assert p[0, 0] == pytest.approx(0.5)
    assert p[0, 1] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert p[0, 2] == pytest.approx(1.0 / (1.0 + math.exp(1.0)))


def test_occupancy_raw_clips_to_byte_range():
    grid = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    out = log_odds.to_occupancy_raw(grid)
    assert out.dtype == np.int8
    assert out[0, 0] == 0
    assert out[0, 1] == 50
    assert out[0, 2] == 100


def test_occupancy_binary_threshold():
    grid = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    out = log_odds.to_occupancy_binary(grid, occ_threshold=0.5)
    assert out.tolist() == [[0, 0, 100]]


def test_apply_scan_marks_endpoint_cell_more_than_traversed():
    grid = np.zeros((20, 20), dtype=np.float32)
    log_odds.apply_scan(
        log_odds=grid,
        ranges=np.array([1.0]),
        angles=np.array([0.0]),
        sensor_x=0.5,
        sensor_y=0.5,
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        max_range=8.0,
        min_range=0.05,
        l_free=-0.4,
        l_occ=0.85,
        l_clamp=5.0,
        sensor_noise_sigma=0.02,
    )
    # Sensor cell traversed first (free); endpoint cell hit (occupied).
    assert grid[5, 5] < 0.0
    # The endpoint cell at (x=1.5, y=0.5) -> grid (5, 15) gets the Gaussian peak.
    assert grid[5, 15] > 0.0


def test_apply_scan_clamps_to_l_clamp():
    grid = np.full((10, 10), 10.0, dtype=np.float32)
    log_odds.apply_scan(
        log_odds=grid,
        ranges=np.array([0.5]),
        angles=np.array([0.0]),
        sensor_x=0.05,
        sensor_y=0.05,
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        max_range=8.0,
        min_range=0.05,
        l_free=-0.4,
        l_occ=0.85,
        l_clamp=2.0,
        sensor_noise_sigma=0.02,
    )
    assert grid.max() <= 2.0 + 1e-6
    assert grid.min() >= -2.0 - 1e-6


def test_apply_scan_skips_invalid_ranges():
    grid = np.zeros((10, 10), dtype=np.float32)
    log_odds.apply_scan(
        log_odds=grid,
        ranges=np.array([float('nan'), 0.001]),
        angles=np.array([0.0, 0.0]),
        sensor_x=0.5,
        sensor_y=0.5,
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        max_range=8.0,
        min_range=0.05,
        l_free=-0.4,
        l_occ=0.85,
        l_clamp=5.0,
        sensor_noise_sigma=0.02,
    )
    # NaN beam and below-min-range beam both skipped; grid unchanged.
    assert np.all(grid == 0.0)
