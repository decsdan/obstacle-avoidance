"""Log-odds inverse sensor model with Gaussian endpoint probability.

The grid stores per-cell log-odds values. Each LIDAR beam contributes a
Bayesian update: cells clearly in front of the hit get a free-space
increment, cells near the hit get an occupancy increment weighted by a
Gaussian on the range-noise sigma, cells beyond the hit are untouched.
Values are clipped to prevent saturation and decay toward zero ("unknown")
over time so that transient obstacles eventually fade.
"""

import math

import numpy as np

from obstacle_grid.amanatides_woo import trace_ray


ENDPOINT_GAUSSIAN_WINDOW = 3.0


def apply_scan(
    log_odds: np.ndarray,
    ranges: np.ndarray,
    angles: np.ndarray,
    sensor_x: float,
    sensor_y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    max_range: float,
    min_range: float,
    l_free: float,
    l_occ: float,
    l_clamp: float,
    sensor_noise_sigma: float,
):
    """Update log-odds grid from one LIDAR scan (in place).

    Args:
        log_odds: ``(H, W)`` float32 log-odds grid, mutated in place.
        ranges: ``(N,)`` beam ranges in meters.
        angles: ``(N,)`` beam angles in the world frame (radians).
        sensor_x, sensor_y: sensor position in world meters.
        origin_x, origin_y: grid origin in world meters.
        resolution: grid resolution (m / cell).
        max_range: beams at or beyond this are treated as no-hit clears.
        min_range: beams shorter than this are discarded as invalid.
        l_free: negative log-odds increment for "observed free".
        l_occ: positive log-odds increment for "observed occupied" at the
            peak of the Gaussian endpoint model.
        l_clamp: absolute clip applied after the pass.
        sensor_noise_sigma: range-noise sigma in meters; endpoint hits
            spread as a Gaussian of this width along the beam.
    """
    h, w = log_odds.shape
    sigma = max(sensor_noise_sigma, 1e-6)
    gate = ENDPOINT_GAUSSIAN_WINDOW * sigma

    for i in range(len(ranges)):
        r = float(ranges[i])
        if not np.isfinite(r) or r < min_range:
            continue

        hit = r < max_range
        r_eff = min(r, max_range)
        if r_eff <= 0.0:
            continue

        a = float(angles[i])
        ex = sensor_x + r_eff * math.cos(a)
        ey = sensor_y + r_eff * math.sin(a)

        for gx, gy, t0, t1 in trace_ray(
                sensor_x, sensor_y, ex, ey,
                origin_x, origin_y, resolution):
            if not (0 <= gx < w and 0 <= gy < h):
                continue

            d_mid = 0.5 * (t0 + t1) * r_eff

            if not hit:
                log_odds[gy, gx] += l_free
                continue

            delta = d_mid - r
            if delta < -gate:
                log_odds[gy, gx] += l_free
            elif delta <= gate:
                weight = math.exp(-0.5 * (delta / sigma) ** 2)
                log_odds[gy, gx] += l_occ * weight
            # delta > gate: beyond the hit, unknown; no update.

    np.clip(log_odds, -l_clamp, l_clamp, out=log_odds)


def decay(log_odds: np.ndarray, dt: float, tau: float):
    """Exponentially pull log-odds toward zero.

    Values closer to zero represent "unknown"; a non-reobserved cell
    loses evidence and eventually returns to the unknown prior.
    """
    if tau <= 0.0 or dt <= 0.0:
        return
    factor = math.exp(-dt / tau)
    log_odds *= factor


def to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds to probability in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-log_odds))


def to_occupancy_raw(log_odds: np.ndarray) -> np.ndarray:
    """Pack log-odds as a 0-100 OccupancyGrid ``int8`` array (probability %)."""
    prob = to_probability(log_odds)
    return np.clip((prob * 100.0).astype(np.int16), 0, 100).astype(np.int8)


def to_occupancy_binary(log_odds: np.ndarray, occ_threshold: float) -> np.ndarray:
    """Threshold log-odds to 0/100 for planner inflation input."""
    return np.where(log_odds > occ_threshold, 100, 0).astype(np.int8)
