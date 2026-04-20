"""Amanatides-Woo voxel traversal."""
import math


def trace_ray(
    x0: float, y0: float,
    x1: float, y1: float,
    origin_x: float, origin_y: float,
    resolution: float,
):
    """Yield ``(gx, gy, t_entry, t_exit)`` for each cell the ray crosses.

    ``t_entry`` and ``t_exit`` are parametric distances in ``[0, 1]`` where
    ``t=0`` is ``(x0, y0)`` and ``t=1`` is ``(x1, y1)``. The ray endpoint
    cell is included; degenerate zero-length rays yield a single cell.
    """
    fx0 = (x0 - origin_x) / resolution
    fy0 = (y0 - origin_y) / resolution
    fx1 = (x1 - origin_x) / resolution
    fy1 = (y1 - origin_y) / resolution

    dx = fx1 - fx0
    dy = fy1 - fy0

    gx = int(math.floor(fx0))
    gy = int(math.floor(fy0))
    gx_end = int(math.floor(fx1))
    gy_end = int(math.floor(fy1))

    if dx == 0.0 and dy == 0.0:
        yield (gx, gy, 0.0, 1.0)
        return

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    if step_x != 0:
        next_bx = gx + (1 if step_x > 0 else 0)
        t_max_x = (next_bx - fx0) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = math.inf
        t_delta_x = math.inf

    if step_y != 0:
        next_by = gy + (1 if step_y > 0 else 0)
        t_max_y = (next_by - fy0) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = math.inf
        t_delta_y = math.inf

    t_prev = 0.0
    while True:
        if gx == gx_end and gy == gy_end:
            yield (gx, gy, t_prev, 1.0)
            return
        if t_max_x < t_max_y:
            t_next = t_max_x
            yield (gx, gy, t_prev, min(t_next, 1.0))
            if t_next >= 1.0:
                return
            gx += step_x
            t_prev = t_next
            t_max_x += t_delta_x
        else:
            t_next = t_max_y
            yield (gx, gy, t_prev, min(t_next, 1.0))
            if t_next >= 1.0:
                return
            gy += step_y
            t_prev = t_next
            t_max_y += t_delta_y
