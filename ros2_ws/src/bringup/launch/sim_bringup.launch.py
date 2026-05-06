"""Top-level sim launch: Gazebo + nav stack + safety_mux + scenario_runner.


* ``use_sim_time`` defaults true and is forwarded to every node.
* The world-layout subordinate seed is forwarded to Gazebo's ``--seed``
  via the ``world_seed`` arg; scenario_runner derives it from the
  master scenario seed before invoking this launch.
* Physics step is fixed in the SDF (1 ms, real-time-factor 0).

The Gazebo process itself is launched via ``ros_gz_sim``; this file
assumes ``ros_gz_sim`` is installed but does not hard-fail if missing,
so the same launch can run the nav stack alone against a pre-running
Gazebo on a different machine.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    ns = LaunchConfiguration('namespace').perform(context)
    world = LaunchConfiguration('world').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    map_yaml = LaunchConfiguration('map_yaml').perform(context)
    global_planner = LaunchConfiguration('global_planner').perform(context)
    local_planner = LaunchConfiguration('local_planner').perform(context)
    world_seed = LaunchConfiguration('world_seed').perform(context)
    launch_gazebo = LaunchConfiguration('gazebo').perform(context).lower() == 'true'

    sim_time_param = {'use_sim_time': use_sim_time.lower() == 'true'}

    actions = [
        LogInfo(msg=f'sim_bringup: world={world} ns={ns} '
                    f'use_sim_time={use_sim_time} world_seed={world_seed}'),
    ]

    if launch_gazebo:
        try:
            ros_gz_share = get_package_share_directory('ros_gz_sim')
            world_path = os.path.join(
                get_package_share_directory('bringup'), 'worlds', f'{world}.sdf')
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    ros_gz_share, 'launch', 'gz_sim.launch.py')),
                launch_arguments={
                    'gz_args': f'-r -s --seed {world_seed} {world_path}',
                }.items(),
            ))
        except Exception as exc:
            actions.append(LogInfo(
                msg=f'ros_gz_sim unavailable; skipping Gazebo launch ({exc})'))

    nav_share = get_package_share_directory('nav_server')
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'namespace': ns,
            'global_planner': global_planner,
            'local_planner': local_planner,
            'map_yaml': map_yaml,
            'use_sim_time': use_sim_time,
            'rviz': LaunchConfiguration('rviz').perform(context),
        }.items(),
    ))

    safety_share = get_package_share_directory('safety_mux')
    safety_params = os.path.join(
        safety_share, 'config', 'safety_mux_params.yaml')
    actions.append(Node(
        package='safety_mux',
        executable='safety_mux',
        name='safety_mux',
        parameters=[safety_params, sim_time_param],
        arguments=['--ros-args', '-p', f'namespace:={ns}'],
        output='screen',
    ))
    actions.append(Node(
        package='safety_mux',
        executable='hazard_adapter',
        name='hazard_adapter',
        parameters=[sim_time_param],
        arguments=['--ros-args', '-p', f'namespace:={ns}'],
        output='screen',
    ))

    runner_share = get_package_share_directory('scenario_runner')
    runner_params = os.path.join(
        runner_share, 'config', 'scenario_runner_params.yaml')
    actions.append(Node(
        package='scenario_runner',
        executable='scenario_runner',
        name='scenario_runner',
        parameters=[runner_params, sim_time_param],
        arguments=['--ros-args', '-p', f'namespace:={ns}'],
        output='screen',
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='/don'),
        DeclareLaunchArgument('world', default_value='W1_indoor_office'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map_yaml', default_value=''),
        DeclareLaunchArgument('global_planner', default_value='a_star'),
        DeclareLaunchArgument('local_planner', default_value='dwa'),
        DeclareLaunchArgument(
            'world_seed', default_value='3',
            description='Gazebo --seed; supply the world_layout subordinate seed'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gazebo', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
