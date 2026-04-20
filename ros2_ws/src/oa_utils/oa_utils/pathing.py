import numpy as np
from scipy.interpolate import CubicSpline


def simplify_path(path: list[tuple[float, float]], max_points: int, epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification + cubic spline smoothing."""
    if len(path) <= 2:
        return path

    rdp_path = rdp(path, epsilon)
    if len(rdp_path) >= 4:
        smoothed = smooth_path(rdp_path, max_points)
    else:
        smoothed = list(rdp_path)

    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker line simplification."""
    if len(points) <= 2:
        return list(points)

    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return [points[0], points[-1]]

    line_unit = line_vec / line_len
    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        proj = np.dot(pt - start, line_unit)
        proj = np.clip(proj, 0, line_len)
        closest = start + proj * line_unit
        dist = np.linalg.norm(pt - closest)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = rdp(points[:max_idx + 1], epsilon)
        right = rdp(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def smooth_path(waypoints: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    """Parametric cubic-spline smoothing, resampled at even intervals."""
    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)
    chord_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    t = np.concatenate([[0], np.cumsum(chord_lengths)])
    total_length = t[-1]

    if total_length < 1e-10:
        return list(waypoints)

    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])

    n_out = min(max_points, max(len(waypoints), 10))
    t_new = np.linspace(0, total_length, n_out)
    return list(zip(cs_x(t_new).tolist(), cs_y(t_new).tolist()))
