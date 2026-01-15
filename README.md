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
Basic launch:
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```

Launch with maze world and SLAM:
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py slam:=true localization:=true world:=maze
```

Launch with maze world and SLAM with Prebuilt Map (mostly so you can see where you are actually putting the goal):
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py slam:=true localization:=true world:=maze map:=$PWD/maze_slamed.yaml
```

### dwa_package - Obstacle avoidance

**Terminal 1:** Launch simulation with SLAM
```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py slam:=true localization:=true world:=maze
```
**Terminal 2:** Run DWA node
```bash
ros2 run dwa_package dwa_node
```


**Terminal 2:** Launch RViz2
```bash
rviz2
```
Set up RViz2, let it load, undock the TurtleBot, then use the 2D Goal Pose button to set goals (only the start of the vector matters, the direction of the arrow is not important)



#### Current DWA Parameters

Parameters can be adjusted via command line using `--ros-args -p parameter:=value`.

In the middle of figuring out the best params, so these are subject to change, but the list below includes what seemed to work the best

**Cost Weights** (tune navigation behavior):

Note: all of the values being multiplied by the weights have been normalized to scale between 0 and 1, meaning the weights are comparable in scale

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights.goal` | 0.3 | Reward for progress toward goal |
| `weights.heading` | 0.02 | Reward for pointing at goal |
| `weights.velocity` | 0.3 | Reward for higher speeds |
| `weights.smoothness` | 0.02 | Reward for less angular velocity |
| `weights.obstacle` | 0.1 | Incentivizes being further from objects |

**Safety Distances** (meters from robot center):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `critical_radius` | 0.22 | Hard rejection boundary (TurtleBot4 radius + 2in) |
| `safe_distance` | 0.35 | Distance for max obstacle score|
| `emergency_stop_distance` | 0.20 | Triggers immediate stop |

**Velocity Limits**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_velocity` | 1.5 | Maximum linear velocity (m/s) |
| `max_angular_velocity` | 2.5 | Maximum angular velocity (rad/s) |

**Planning**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_steps` | 30 | Trajectory lookahead steps |
| `dt` | 0.1 | Time step (s), total lookahead = steps × dt |
| `goal_tolerance` | 0.3 | Distance to consider goal reached (m) |

**Example:** Run with custom weights for aggressive obstacle avoidance:

```bash
ros2 run dwa_package dwa_node --ros-args \
    -p weights.goal:=0.4 \
    -p weights.obstacle:=0.5 \
    -p max_velocity:=1.0
```