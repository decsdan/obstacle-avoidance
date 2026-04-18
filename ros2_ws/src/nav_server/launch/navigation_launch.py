"""Unified navigation launch file for the obstacle-avoidance fork.

Launches any combination of global planner, local planner, navigation
server, and RViz from a single terminal. Generates a namespace-aware
RViz config with the correct displays for the selected planners.

AMCL localization must be running separately to provide the map -> odom
transform. After launch, send goals via the RViz '2D Goal Pose' tool or
``ros2 run nav_server navigate -- --goal X Y``.

Common Commands:

    # A* + DWA stacked (default)
    ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/map.yaml

    # A* standalone (no local planner)
    ros2 launch nav_server navigation_launch.py \\
        map_yaml:=/path/to/map.yaml local_planner:=none

    # DWA standalone (LIDAR-only reactive navigation)
    ros2 launch nav_server navigation_launch.py \\
        global_planner:=none local_planner:=dwa

    # Custom namespace, no RViz
    ros2 launch nav_server navigation_launch.py \\
        map_yaml:=/path/to/map.yaml namespace:=/robot rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context):
    """Resolve launch args and emit the conditional set of nodes."""
    ns = LaunchConfiguration('namespace').perform(context)
    global_planner = LaunchConfiguration('global_planner').perform(context)
    local_planner = LaunchConfiguration('local_planner').perform(context)
    map_yaml = LaunchConfiguration('map_yaml').perform(context)
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() == 'true'
    use_server = LaunchConfiguration('server').perform(context).lower() == 'true'

    dwa_pkg_dir = get_package_share_directory('dwa_package')
    default_params = os.path.join(dwa_pkg_dir, 'config', 'stacked_params.yaml')
    params_file = LaunchConfiguration('params_file').perform(context)
    if not params_file:
        params_file = default_params

    nav_pkg_dir = get_package_share_directory('nav_server')
    nav_params = os.path.join(nav_pkg_dir, 'config', 'nav_server_params.yaml')

    has_global = global_planner not in ('none', '')
    has_local = local_planner not in ('none', '')
    stacked = has_global and has_local
    stacked_str = 'true' if stacked else 'false'

    actions = []

    mode_str = (f'{global_planner} + {local_planner}' if stacked
                else (global_planner if has_global else local_planner))
    actions.append(LogInfo(
        msg=f'Navigation launch: {mode_str} | namespace={ns} | rviz={use_rviz}'))

    if has_global and map_yaml:
        actions.append(SetEnvironmentVariable('MAP_YAML', map_yaml))

    needs_obstacle_grid = (
        local_planner == 'dwa' or global_planner == 'd_star')
    if needs_obstacle_grid:
        actions.append(Node(
            package='obstacle_grid',
            executable='obstacle_grid',
            name='obstacle_grid_node',
            arguments=['--ros-args', '-p', f'namespace:={ns}'],
            output='screen',
        ))

    global_nodes = {
        'a_star': ('a_star', 'a_star_nav', 'a_star_nav', True),
        'd_star': ('d_star', 'd_star_nav', 'd_star_nav', False),
        'jps': ('jps', 'jps_nav', 'jps_nav', False),
    }
    if global_planner in global_nodes:
        pkg, executable, name, uses_params = global_nodes[global_planner]
        node_kwargs = {
            'package': pkg,
            'executable': executable,
            'name': name,
            'arguments': [
                '--ros-args',
                '-p', f'namespace:={ns}',
                '-p', f'stacked:={stacked_str}',
            ],
            'output': 'screen',
        }
        if uses_params:
            node_kwargs['parameters'] = [params_file]
        actions.append(Node(**node_kwargs))

    if local_planner == 'dwa':
        actions.append(Node(
            package='dwa_package',
            executable='dwa_node',
            name='dwa_node',
            parameters=[params_file],
            arguments=[
                '--ros-args',
                '-p', f'namespace:={ns}',
                '-p', f'stacked:={stacked_str}',
            ],
            output='screen',
        ))

    if use_server:
        server_overrides = [
            '--ros-args',
            '-p', f'namespace:={ns}',
        ]
        if has_global:
            server_overrides += [
                '-p', f'default_global_planner:={global_planner}',
            ]
        if has_local:
            server_overrides += [
                '-p', f'default_local_planner:={local_planner}',
            ]
        actions.append(Node(
            package='nav_server',
            executable='navigation_server',
            name='navigation_server',
            parameters=[nav_params],
            arguments=server_overrides,
            output='screen',
        ))

    if use_rviz:
        from nav_server.rviz_config import write_rviz_config
        rviz_path = write_rviz_config(ns, global_planner, local_planner)

        actions.append(LogInfo(msg=f'RViz config written to: {rviz_path}'))
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', rviz_path,
                '--ros-args',
                '-r', f'/tf:={ns}/tf',
                '-r', f'/tf_static:={ns}/tf_static',
            ],
            output='screen',
        ))

    return actions


def generate_launch_description():
    dwa_pkg_dir = get_package_share_directory('dwa_package')
    default_params = os.path.join(dwa_pkg_dir, 'config', 'stacked_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_yaml', default_value='',
            description='Path to map YAML file (required for A*, D*, JPS)'),
        DeclareLaunchArgument(
            'namespace', default_value='/don',
            description='Robot namespace'),
        DeclareLaunchArgument(
            'global_planner', default_value='a_star',
            description=('Global planner: a_star, d_star, jps, rrt_star, fm2, '
                         'or none (skip global launch)')),
        DeclareLaunchArgument(
            'local_planner', default_value='dwa',
            description=('Local planner: dwa, mppi, rl_policy, or none '
                         '(skip local launch). nav_server still requires a '
                         'valid local for the Navigate contract.')),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch RViz with auto-configured displays'),
        DeclareLaunchArgument(
            'server', default_value='true',
            description='Launch the navigation action server'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Path to shared parameter YAML file'),

        OpaqueFunction(function=_launch_setup),
    ])
