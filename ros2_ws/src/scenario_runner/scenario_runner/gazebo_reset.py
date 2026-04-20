"""Gazebo world-reset and entity-teleport service clients."""
import math
from typing import Optional

import rclpy
from rclpy.node import Node

try:
    from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
    _GZ_IFACE_AVAILABLE = True
except ImportError:
    _GZ_IFACE_AVAILABLE = False


class GazeboResetClient:
    """Thin wrapper around Gazebo reset + teleport services."""

    def __init__(self, node: Node, world_name: str):
        if not _GZ_IFACE_AVAILABLE:
            raise RuntimeError(
                'ros_gz_interfaces not available; cannot drive Gazebo reset')
        self._node = node
        self._reset_cli = node.create_client(
            ControlWorld, f'/world/{world_name}/control')
        self._pose_cli = node.create_client(
            SetEntityPose, f'/world/{world_name}/set_pose')
        self._world_name = world_name

    def wait_for_services(self, timeout_sec: float) -> bool:
        return (
            self._reset_cli.wait_for_service(timeout_sec=timeout_sec)
            and self._pose_cli.wait_for_service(timeout_sec=timeout_sec)
        )

    def reset_world(self, timeout_sec: float = 5.0) -> bool:
        """Issue ``reset: true`` on the world control service."""
        req = ControlWorld.Request()
        req.world_control.pause = False
        req.world_control.reset.all = True
        future = self._reset_cli.call_async(req)
        return self._spin_future(future, timeout_sec)

    def teleport(
        self,
        entity_name: str,
        x: float,
        y: float,
        theta: float,
        timeout_sec: float = 5.0,
    ) -> bool:
        req = SetEntityPose.Request()
        req.entity.name = entity_name
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = 0.0
        # 2D yaw -> quaternion (z, w)
        half = 0.5 * float(theta)
        req.pose.orientation.z = float(math.sin(half))
        req.pose.orientation.w = float(math.cos(half))
        future = self._pose_cli.call_async(req)
        return self._spin_future(future, timeout_sec)

    def _spin_future(self, future, timeout_sec: float) -> bool:
        rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=timeout_sec)
        if not future.done():
            return False
        result = future.result()
        return bool(getattr(result, 'success', True))


def is_available() -> bool:
    """True when ros_gz_interfaces is importable in this environment."""
    return _GZ_IFACE_AVAILABLE
