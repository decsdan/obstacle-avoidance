"""Distance-transform utilities for occupancy grid inflation."""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def inflate_binary(binary_grid: np.ndarray, inflation_cells: int) -> np.ndarray:
    """Circular dilation of a 0/100 binary occupancy mask."""
    if inflation_cells <= 0:
        return binary_grid.copy()
    y, x = np.ogrid[-inflation_cells:inflation_cells + 1,
                    -inflation_cells:inflation_cells + 1]
    kernel = (x ** 2 + y ** 2) <= inflation_cells ** 2
    obstacle_mask = (binary_grid >= 100)
    inflated = binary_dilation(obstacle_mask, structure=kernel)
    return np.where(inflated, 100, 0).astype(np.int8)


def distance_field(binary_grid: np.ndarray, resolution: float) -> np.ndarray:
    """Euclidean distance to nearest obstacle (meters)."""
    free = binary_grid < 100
    return distance_transform_edt(free).astype(np.float32) * resolution
