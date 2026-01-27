# Obstacle Avoidance COMPS

ROS 2 workspace for TurtleBot4 obstacle avoidance and navigation.

> **Note:** This repository supports both ROS 2 Humble and Jazzy. Replace `humble` with `jazzy` in commands based on your distribution. Check with: `echo $ROS_DISTRO`

---

## Initial Setup

### 1. Build the Workspace

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build
```

### 2. Configure Your Environment

Add these lines to your `~/.bashrc`:

```bash
# ROS 2 Setup
source /opt/ros/jazzy/setup.bash  # or 'humble'

# This is for running the physical robot
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start
```
### 3. Manual Control on Physical Robot
```bash
# Note: ROBOT_NAME must match robot /cmd_vel:=/{ROBOT_NAME}/cmd_vel
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true -r /cmd_vel:=/don/cmd_vel

```

### 3. Running Slam on Physical Robot

Open two terminals and follow the previous step (configure) in each.

In terminal 1 run:

```bash
ros2 launch turtlebot4_navigation slam.launch.py namespace:=/don
```

In terminal 2 run:

```bash
ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/don
```

then move the robot around to start.

Apply changes:
```bash
source ~/.bashrc
```

---

## Running Gazebo Simulation

### Basic Gazebo Launch

```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```

### With SLAM and Navigation

```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py slam:=true nav2:=true rviz:=true world:=maze
```

---

## Running Custom Packages

### Demo Package
Makes the robot move in a circle.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select demo_package
source install/local_setup.bash
ros2 run demo_package demo_node
```

### Joy Test
Prints the robot's position data in the world.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --symlink-install
source install/local_setup.bash
ros2 run joy_test input
```

### Sensor Data
Displays GUI for robot's camera or lidar.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --symlink-install
source install/local_setup.bash

# Camera view
ros2 run sensor_data camera

# Lidar view
ros2 run sensor_data lidar
```

### A* Navigator
Autonomous navigation using A* pathfinding algorithm with hybrid obstacle checking.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select a_star
source install/local_setup.bash

# Run A* navigator (prompts for goal coordinates)
ros2 run a_star a_star_nav

# With custom safety parameters
ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run a_star a_star_nav

# Run interactive visualizer
ros2 run a_star visualizer
```

**Features:**
- Interactive goal input via terminal
- Hybrid obstacle validation (tight spaces + safety margins)
- Pure pursuit waypoint following
- Environment variable configuration

### D* Lite Navigator
Autonomous navigation using D* Lite with dynamic replanning and obstacle detection.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select d_star
source install/local_setup.bash

# Run D* Lite navigator (prompts for goal coordinates)
ros2 run d_star d_star_nav

# With custom safety parameters
ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.15 ros2 run d_star d_star_nav

# Run interactive visualizer with obstacle placement
ros2 run d_star visualizer

# Run live path visualizer (shows planned vs traveled path)
ros2 run d_star live_visualizer
```

**Features:**
- Incremental pathfinding with dynamic replanning
- SLAM integration for obstacle discovery
- Lidar-based dynamic obstacle detection
- Automatic replanning when obstacles detected
- Real-time path visualization

---
### Jump Point Search (JPS) Navigator
Autonomous navigation using Jump Point Search algorithm with optimized pathfinding through uniform-cost grids.




### DWA Navigator

These instructions assume robot namespace `/don`. Replace with your robot's namespace if different.

#### Terminal 1 - DWA Node
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select dwa_package
source install/setup.bash

ros2 run dwa_package dwa_node
```

#### Terminal 2 - RViz2 Visualization
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/don
```

Then in RViz: File → Open Config → `~/obstacle-avoidance-comps/don_viz.rviz`

#### RViz2 Configuration

**Add 2D Goal Pose tool** (required for sending goals):
1. Panels → Add New Tool → select "2D Goal Pose" → OK
2. Use the new toolbar button to click+drag goals in the map

**Add DWA visualization topics:**
1. Add → By topic → select the following:
   - `/don/scan` (LaserScan) - LiDAR points
   - `/don/debug_obstacles` (Marker) - obstacle XY coordinates as red spheres
   - `/don/dwa/trajectories` (Marker) - candidate paths (red→green by score)
   - `/don/dwa/best_trajectory` (Marker) - selected path (cyan)

**Save configuration:**
1. File → Save Config As → `~/obstacle-avoidance-comps/don_viz.rviz`


#### Terminal 3 - Debugging (optional)
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash

ros2 topic hz /don/scan
ros2 topic echo /don/goal_pose
ros2 topic echo /don/cmd_vel
```

#### DWA Parameters

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























```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select jps
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash

# Run JPS navigator (gazebo prompts for goal coordinates)
ros2 run jps jps_nav

# With custom safety parameters
ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run jps jps_nav

# Run interactive visualizer
ros2 run jps jps_visualizer

# Monitor robot position
ros2 run jps odom
```

**Features:**
- Interactive goal input via terminal
- Jump Point Search optimization for uniform-cost grids
- Hybrid obstacle validation (tight spaces + safety margins)
```

---
## Using Saved Maps with Localization and Navigation

### Prerequisites

Install required packages:
```bash
sudo apt install ros-${ROS_DISTRO}-nav2-map-server
```



### Launch with Pre-Saved Map

If you have a saved map (e.g., `maze_slamed.pgm` and `maze_slamed.yaml`):

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
source install/setup.bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  slam:=false \
  localization:=true \
  nav2:=true \
  rviz:=true \
  world:=maze \
  map:=$PWD/maze_slamed.yaml
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `slam:=false` | Don't create a new map (using saved map) |
| `localization:=true` | Use AMCL to localize on existing map |
| `nav2:=true` | Enable Nav2 for autonomous navigation |
| `rviz:=true` | Launch RViz for visualization |
| `world:=maze` | Load the maze world in Gazebo |
| `map:=<path>` | Path to your saved map YAML file |

### RViz Configuration

1. **Fixed Frame** should automatically be set to `map`
2. Your saved map should be visible
3. Click **2D Pose Estimate** button (top toolbar)
4. Click and drag on the map where the robot is in Gazebo to set initial pose
5. Once localized, use **Nav2 Goal** button to send navigation goals

> **Important:** The robot needs an accurate initial pose for AMCL to work properly. The particle cloud should converge around the robot's actual position.

---

## Project Structure

```
ros2_ws/
├── src/
│   ├── a_star/          # A* pathfinding implementation
│   ├── d_star/          # D* pathfinding implementation
│   ├── demo_package/    # Demo movement package
│   ├── joy_test/        # Position data testing
│   └── sensor_data/     # Camera and lidar visualization
├── maze_slamed.pgm      # Saved maze map (image)
└── maze_slamed.yaml     # Saved maze map (metadata)
```

---

## Troubleshooting

### Map frame not available in RViz
- Ensure localization is enabled with `localization:=true`
- Check that map path is correct and absolute
- Verify AMCL is running: `ros2 node list | grep amcl`

### Robot not localizing properly
- Set accurate initial pose using **2D Pose Estimate** in RViz
- Ensure the map matches the Gazebo world
- Check particle cloud convergence

### Commands not found
- Make sure you've sourced the workspace: `source install/setup.bash`
- Verify ROS 2 is sourced: `echo $ROS_DISTRO`