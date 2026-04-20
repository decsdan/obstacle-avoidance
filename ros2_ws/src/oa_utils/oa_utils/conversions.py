import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def grid_from_msg(grid_msg, occ_threshold: int = 50) -> np.ndarray:
    """Convert a ROS OccupancyGrid into a 2D numpy array (0 = free, 1 = blocked).

    Cells with occupancy probability >= occ_threshold or unknown (-1) are
    treated as blocked. Origin orientation is ignored (assumed axis-aligned).
    """
    width = grid_msg.info.width
    height = grid_msg.info.height
    data = np.array(grid_msg.data, dtype=np.int16).reshape((height, width))
    grid = np.zeros_like(data, dtype=np.uint8)
    grid[(data >= occ_threshold) | (data < 0)] = 1
    return grid


def world_to_grid(x: float, y: float, resolution: float, origin: tuple[float, float]) -> tuple[int, int]:
    gx = int((x - origin[0]) / resolution)
    gy = int((y - origin[1]) / resolution)
    return (gx, gy)


def grid_to_world(gx: int, gy: int, resolution: float, origin: tuple[float, float]) -> tuple[float, float]:
    x = (gx + 0.5) * resolution + origin[0]
    y = (gy + 0.5) * resolution + origin[1]
    return (x, y)


def build_path_msg(waypoints: list[tuple[float, float]], frame_id: str, stamp) -> Path:
    """Construct a nav_msgs/Path from a list of (x, y) world coordinates."""
    path_msg = Path()
    path_msg.header.stamp = stamp
    path_msg.header.frame_id = frame_id

    yaw = 0.0
    for i, (wx, wy) in enumerate(waypoints):
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = float(wx)
        pose.pose.position.y = float(wy)
        pose.pose.position.z = 0.0
        if i + 1 < len(waypoints):
            nx, ny = waypoints[i + 1]
            yaw = math.atan2(ny - wy, nx - wx)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path_msg.poses.append(pose)
    return path_msg
