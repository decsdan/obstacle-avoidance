"""Unified navigation launch file for the obstacle-avoidance fork."""
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


# Global planner registry: name -> (package, executable, node_name, params_file)
# node_name matches what the planner calls itself via super().__init__ so the
# YAML keys in each config file land on the right node.
_GLOBAL_PLANNERS = {
    'a_star': {
        'package': 'a_star',
        'executable': 'a_star_nav',
        'node_name': 'a_star_planner',
        'params': ('a_star', 'a_star_params.yaml'),
    },
    'd_star': {
        'package': 'd_star',
        'executable': 'd_star_nav',
        'node_name': 'd_star_planner',
        'params': ('d_star', 'd_star_params.yaml'),
    },
    'jps': {
        'package': 'jps',
        'executable': 'jps_nav',
        'node_name': 'jps_planner',
        'params': ('jps', 'jps_params.yaml'),
    },
}


def _launch_setup(context):
    """Resolve launch args and emit the conditional set of nodes."""
    ns = LaunchConfiguration('namespace').perform(context)
    global_planner = LaunchConfiguration('global_planner').perform(context)
    local_planner = LaunchConfiguration('local_planner').perform(context)
    map_yaml = LaunchConfiguration('map_yaml').perform(context)
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() == 'true'
    use_server = LaunchConfiguration('server').perform(context).lower() == 'true'

    nav_pkg_dir = get_package_share_directory('nav_server')
    nav_params = os.path.join(nav_pkg_dir, 'config', 'nav_server_params.yaml')

    has_global = global_planner not in ('none', '')
    has_local = local_planner not in ('none', '')

    actions = []

    mode_str = (f'{global_planner} + {local_planner}'
                if (has_global and has_local)
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

    if global_planner in _GLOBAL_PLANNERS:
        spec = _GLOBAL_PLANNERS[global_planner]
        params_pkg, params_name = spec['params']
        params_file = os.path.join(
            get_package_share_directory(params_pkg), 'config', params_name)
        actions.append(Node(
            package=spec['package'],
            executable=spec['executable'],
            name=spec['node_name'],
            parameters=[params_file],
            arguments=['--ros-args', '-p', f'namespace:={ns}'],
            output='screen',
        ))

    if local_planner == 'dwa':
        dwa_params = os.path.join(
            get_package_share_directory('dwa_package'),
            'config', 'dwa_params.yaml')
        actions.append(Node(
            package='dwa_package',
            executable='dwa_node',
            name='dwa_follower',
            parameters=[dwa_params],
            arguments=['--ros-args', '-p', f'namespace:={ns}'],
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
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_yaml', default_value='',
            description='Path to map YAML file (required for A*, D*, JPS)'),
        DeclareLaunchArgument(
            'namespace', default_value='/don',
            description='Robot namespace'),
        DeclareLaunchArgument(
            'global_planner', default_value='a_star',
            description=('Global planner: a_star, d_star, jps, or none '
                         '(skip global launch)')),
        DeclareLaunchArgument(
            'local_planner', default_value='dwa',
            description=('Local planner: dwa, or none (skip local launch). '
                         'nav_server still requires a valid local for the '
                         'Navigate contract.')),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch RViz with auto-configured displays'),
        DeclareLaunchArgument(
            'server', default_value='true',
            description='Launch the navigation action server'),

        OpaqueFunction(function=_launch_setup),
    ])
