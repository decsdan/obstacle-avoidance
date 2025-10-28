# obstacle-avoidance-comps

Lab Computer Setup:

gedit ~/.bashrc

In this file, add these commands:

source /opt/ros/humble/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source ~/obstacle-avoidance-comps/ros2_ws/install/setup.bash

This allows for ROS2, Colcon, and python packages to run.

 


Setup Gazebo (for the lab computers):

Initial setup (needs to run after for each terminal using ros):
$ source /opt/ros/humble/setup.bash

Run Gazebo:
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py

running the ROS2 demo_package:
- cd into ros2_ws package
- run $ rosdep install -i --from-path src --rosdistro humble -y
- then run $ colcon build --packages-select demo_package
- then run $ source install/local_setup.bash
- then run $ ros2 run demo_package demo_node
