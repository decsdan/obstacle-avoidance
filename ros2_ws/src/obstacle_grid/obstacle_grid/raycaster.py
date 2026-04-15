"""Bresenham raycasting with two-tier obstacle clearing.

Tier 1 (raycasting): cells along each ray before the hit are marked free
immediately. Tier 2 (temporal decay): occupied cells not re-confirmed
within decay_seconds are cleared to handle occluded regions and blind spots.
"""

import math

import numpy as np


def bresenham_cells(x0: int, y0: int, x1: int, y1: int):
    """Return all (x, y) grid cells on the line from (x0,y0) to (x1,y1)."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


def process_scan(
    grid: np.ndarray,
    last_occupied: np.ndarray,
    ranges: np.ndarray,
    angles: np.ndarray,
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    max_range: float,
    min_range: float,
    current_time: float,
):
    """Raycast a full LIDAR scan into the grid.

    Marks cells along each ray as free and the endpoint as occupied.
    Modifies grid and last_occupied in-place. Returns set of changed cells.
    """
    height, width = grid.shape
    changed = set()

    sensor_gx = int((sensor_x - origin_x) / resolution)
    sensor_gy = int((sensor_y - origin_y) / resolution)

    for i in range(len(ranges)):
        r = ranges[i]

        if not np.isfinite(r) or r < min_range:
            continue

        hit = r < max_range
        effective_range = min(r, max_range)

        world_angle = sensor_yaw + angles[i]
        end_x = sensor_x + effective_range * math.cos(world_angle)
        end_y = sensor_y + effective_range * math.sin(world_angle)

        end_gx = int((end_x - origin_x) / resolution)
        end_gy = int((end_y - origin_y) / resolution)

        cells = bresenham_cells(sensor_gx, sensor_gy, end_gx, end_gy)

        if not cells:
            continue

        for gx, gy in cells[:-1]:
            if 0 <= gx < width and 0 <= gy < height:
                old = grid[gy, gx]
                grid[gy, gx] = 0
                if old != 0:
                    changed.add((gx, gy))

        last_gx, last_gy = cells[-1]
        if 0 <= last_gx < width and 0 <= last_gy < height:
            old = grid[last_gy, last_gx]
            new_val = 100 if hit else 0
            grid[last_gy, last_gx] = new_val
            if new_val == 100:
                last_occupied[last_gy, last_gx] = current_time
            if old != new_val:
                changed.add((last_gx, last_gy))

    return changed


def apply_temporal_decay(
    grid: np.ndarray,
    last_occupied: np.ndarray,
    current_time: float,
    decay_seconds: float,
):
    """Clear occupied cells not re-confirmed within decay_seconds."""
    stale_mask = (grid == 100) & ((current_time - last_occupied) > decay_seconds)
    cleared = set()

    if stale_mask.any():
        ys, xs = np.where(stale_mask)
        for x, y in zip(xs.tolist(), ys.tolist()):
            cleared.add((x, y))
        grid[stale_mask] = 0

    return cleared


def inflate_grid(grid: np.ndarray, inflation_cells: int) -> np.ndarray:
    """Inflate obstacles by a circular kernel of given radius in cells."""
    if inflation_cells <= 0:
        return grid.copy()

    from scipy.ndimage import binary_dilation

    y, x = np.ogrid[-inflation_cells:inflation_cells+1,
                     -inflation_cells:inflation_cells+1]
    kernel = (x**2 + y**2) <= inflation_cells**2

    obstacle_mask = (grid == 100)
    inflated = binary_dilation(obstacle_mask, structure=kernel)
    result = np.where(inflated, 100, 0).astype(np.int8)
    return result
