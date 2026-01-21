# Obstacle Avoidance - ROS2 Jazzy
Dynamic Window Approach (DWA) for TurtleBot4 in Gazebo Harmonic and physical hardware.

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

## Running in Simulation

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

---

## Running on Physical TurtleBot4

These instructions assume robot namespace `/don`. Replace with your robot's namespace if different.

### Terminal 1 - DWA Node
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select dwa_package
source install/setup.bash

ros2 run dwa_package dwa_node
```

### Terminal 2 - RViz2 Visualization
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/don
```

### RViz2 Configuration

**Add 2D Goal Pose tool** (required for sending goals):
1. Panels → Add New Tool → select "2D Goal Pose" → OK
2. Use the new toolbar button to click+drag goals in the map

**Add DWA visualization topics:**
1. Add → By topic → select the following:
   - `/don/scan` (LaserScan) - LiDAR points
   - `/don/debug_obstacles` (Marker) - obstacle XY coordinates as red spheres
   - `/don/dwa/trajectories` (Marker) - candidate paths (red→green by score)
   - `/don/dwa/best_trajectory` (Marker) - selected path (cyan)

**Save configuration** (optional):
1. File → Save Config As → `~/dwa_nav.rviz`
2. Future launches: `rviz2 -d ~/dwa_nav.rviz`

### Terminal 3 - Debugging (optional)
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash

ros2 topic hz /don/scan
ros2 topic echo /don/goal_pose
ros2 topic echo /don/cmd_vel
```

### Manual Goal Publishing
```bash
ros2 topic pub --once /don/goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'odom'}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

---

## DWA Parameters

Parameters can be adjusted via command line using `--ros-args -p parameter:=value`.

**Cost Weights** (normalized 0-1, comparable scale):

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
| `critical_radius` | 0.30 | Hard rejection boundary |
| `safe_distance` | 0.50 | Distance for max obstacle score |
| `emergency_stop_distance` | 0.25 | Triggers immediate stop |

**Velocity Limits**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_velocity` | 1.0 | Maximum linear velocity (m/s) |
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
    -p max_velocity:=0.5
```