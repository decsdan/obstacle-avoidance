"""Launch the scenario_runner node."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('scenario_runner')
    default_params = os.path.join(
        pkg_share, 'config', 'scenario_runner_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace', default_value='/don',
            description='Robot namespace'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Path to scenario_runner params YAML'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use sim-time clock; keep true for deterministic runs'),
        Node(
            package='scenario_runner',
            executable='scenario_runner',
            name='scenario_runner',
            parameters=[
                LaunchConfiguration('params_file'),
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            arguments=[
                '--ros-args',
                '-p', ['namespace:=', LaunchConfiguration('namespace')],
            ],
            output='screen',
        ),
    ])
