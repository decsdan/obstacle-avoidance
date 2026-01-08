# Obstacle Avoidance - ROS2 Jazzy

Dynamic Window Approach (DWA) for TurtleBot4 in Gazebo Harmonic.

## Quick Install

```bash
# Install dependencies
sudo apt update
sudo apt install ros-jazzy-desktop gz-harmonic ros-jazzy-turtlebot4-simulator

# Setup rosdep (first time only)
sudo rosdep init
rosdep update
```

## Setup

Add to `~/.bashrc`:
```bash
source /opt/ros/jazzy/setup.bash
source ~/obstacle-avoidance-comps/ros2_ws/install/setup.bash
```

Build workspace:
```bash
cd ~/obstacle-avoidance-comps/ros2_ws
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build
source install/setup.bash
```

## Running

### Launch Simulation
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```

### demo_package - Circular motion
```bash
ros2 run demo_package demo_node
```

### joy_test - Position output
```bash
ros2 run joy_test input
```

### sensor_data - Camera/LiDAR
```bash
ros2 run sensor_data camera  # or lidar
```

### dwa_package - Obstacle avoidance

**Terminal 1:**
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```

**Terminal 2:**
```bash
rviz2
```

Set Rviz2 up, let it load, undock the turtlebot, then use the 2D goal Pose tool to create the goal for turtlebot

**Terminal 3:**
```bash
ros2 run dwa_package dwa_node
```
## Key Changes for Jazzy
- `turtlebot4_ignition_bringup` → `turtlebot4_gz_bringup`
- `turtlebot4_ignition.launch.py` → `turtlebot4_gz.launch.py`