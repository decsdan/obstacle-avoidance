# obstacle-avoidance-comps

Lab Computer Setup:

**important**
- cd into ros2_ws package
- run colcon build
  
Then

gedit ~/.bashrc

In this file, add these commands:

source /opt/ros/humble/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source ~/obstacle-avoidance-comps/ros2_ws/install/setup.bash
source install/setup.bash

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

This makes the robot move in a circle.

running the ROS2 joy_test:
- cd into ros2_ws package
- run $ colcon build --symlink-install
- then run $ source install/local_setup.bash
- then run $ ros2 run joy_test input

This prints the position data of where the robot is the in the world.


running the ROS2 sensor_data:
- cd into ros2_ws package
- run $ colcon build --symlink-install
- then run $ source install/local_setup.bash
- then run $ ros2 run sensor_data camera (robot's camera pov)
- or run $ ros2 run sensor_data lidar (robot's lidar pov)

Shows a GUI of either the camera/lidar of the robot.


To Run the DWA package: (which currently doesn't process the message inputs, or publish to twist, will be implementing that within the next few days)

 - download src files
 - cd into ros2_ws package
 - $ source install/local_setup.bash
 - $ ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
 - in another terminal, 
$ colcon build
$ source install/local_setup.bash
$ ros2 run dwa_package dwa_node


This should return info on what functions are being run, in what order (and you can tweak dwa_node.py to also return the actual messages from LaserScan and Odometry)


