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

### 4. Running Slam on Physical Robot

Open two terminals and follow the previous step (configure) in each.

In terminal 1 run:

```bash
ros2 launch turtlebot4_navigation slam.launch.py namespace:=/don
```
### 5. Running Nav2 on Physical Robot

In terminal 2 run:

```bash
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/don
```

### 6. Running Rviz on Physical Robot

In terminal 3 run:

```bash
ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/don
```

then move the robot around to start. This will create a SLAM map that you can save 

---

### 7. Saving Maps for Navigation

To save a SLAM map for use with localization:

```bash
ros2 run nav2_map_server map_saver_cli -f "map_name" \
    --ros-args -p map_subscribe_transient_local:=true -r __ns:=/don
```

This creates `map_name.yaml` and `map_name.pgm` files.

Apply changes:
```bash
source ~/.bashrc
```

---

## Running with AMCL Localization

This workflow uses a pre-saved map with AMCL localization, allowing you to run multiple navigation experiments without restarting terminals.

### Prerequisites

Install required packages:
```bash
sudo apt install ros-${ROS_DISTRO}-nav2-map-server ros-${ROS_DISTRO}-nav2-amcl
```

This workspace includes `amcl_params.yaml` which configures AMCL to auto-set an initial pose at (0,0,0) on startup. This is required because the `map` frame won't exist in RViz until AMCL initializes, and without it you can't see the map to place a 2D Pose Estimate. The auto-pose bootstraps the `map` frame so RViz works immediately — you then refine with 2D Pose Estimate.

### Experiment Workflow (Physical Robot)

**Terminal 1 - Localization (AMCL):**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

# Launch localization with your saved map and AMCL params
ros2 launch turtlebot4_navigation localization.launch.py \
    namespace:=/don \
    map:=/path/to/your/map.yaml \
    params_file:=/path/to/obstacle-avoidance-comps/ros2_ws/amcl_params.yaml
```

Wait for AMCL to print `Setting pose` — the `map` frame is now available.

If AMCL is still stuck on "Please set the initial pose...", the params file may not have loaded. See **Manual Initial Pose Fallback** below.

**Terminal 2 - RViz Visualization:**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash

ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/don
```

**Terminal 3 - Navigation Node (A* or DWA):**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
cd ~/obstacle-avoidance-comps/ros2_ws
source install/setup.bash

# For A* navigation:
ros2 run a_star a_star_nav

# Or for DWA navigation:
ros2 run dwa_package dwa_node
```

### Manual Initial Pose Fallback

If `amcl_params.yaml` doesn't auto-set the initial pose (AMCL keeps printing "Please set the initial pose..."), bootstrap it manually. Run this in a spare terminal and wait for AMCL to print `Setting pose`, then Ctrl+C:

```bash
ros2 topic pub -r 1 /don/initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: 0.0, y: 0.0, z: 0.0},
      orientation: {w: 1.0}
    }
  }
}"
```

> **Note:** Use `-r 1` (repeated), not `--once`. The single-shot publish often fires before AMCL discovers the publisher and gets silently dropped.

Once the `map` frame exists, RViz will display the map and you can refine with **2D Pose Estimate** as usual.

### Running Repeated Experiments

Once all terminals are running, you can run multiple experiments without restarting:

1. **Set Initial Pose (required after moving robot):**
   - Ensure the 2D Pose Estimate topic is set to `/don/initialpose` (Panels → Tool Properties)
   - In RViz, click the **"2D Pose Estimate"** button in the top toolbar
   - Click and drag on the map where the robot actually is
   - The arrow shows the robot's heading direction
   - Watch the particle cloud (red arrows) converge around the robot

2. **Send Goal:**
   - Click the **"2D Goal Pose"** button in RViz toolbar
   - Click and drag on the map where you want the robot to go
   - The navigator will plan and execute the path

3. **After Goal Reached:**
   - Manually move/lift robot back to starting position
   - Set initial pose again using "2D Pose Estimate"
   - Send a new goal
   - Repeat as needed!

### RViz Setup for Navigation

**Configure 2D Pose Estimate tool:**
1. In RViz: Panels → Tool Properties
2. Find "2D Pose Estimate" tool
3. Set Topic to: `/{namespace}/initialpose` (e.g., `/don/initialpose`)

**Configure 2D Goal Pose tool:**
1. In RViz: Panels → Tool Properties
2. Find "2D Goal Pose" tool
3. Set Topic to: `/{namespace}/goal_pose` (e.g., `/don/goal_pose`)

**Add visualization topics:**
- `/don/map` (Map) - Occupancy grid from localization. If it shows "no map received", expand the display properties and set **Durability** to **Transient Local**
- `/don/scan` (LaserScan) - LiDAR points
- `/don/particlecloud` (PoseArray) - AMCL particle cloud

**For DWA visualization, also add:**
- `/don/debug_obstacles` (Marker) - Detected obstacles
- `/don/dwa/trajectories` (Marker) - Candidate paths
- `/don/dwa/best_trajectory` (Marker) - Selected path

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

### Using Saved Map in Simulation

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
source install/setup.bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  slam:=false \
  localization:=true \
  nav2:=false \
  rviz:=true \
  world:=maze \
  map:=$PWD/maze_slamed.yaml
```

Then in a separate terminal, run your navigation node.

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
Autonomous navigation using A* pathfinding algorithm with TF2/AMCL localization.

```bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build --packages-select a_star
source install/local_setup.bash

# Run A* navigator (receives goals via RViz 2D Goal Pose)
ros2 run a_star a_star_nav

# With custom safety parameters
ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run a_star a_star_nav

# Run interactive visualizer
ros2 run a_star visualizer
```

**Features:**
- Receives goals via RViz "2D Goal Pose" tool
- Uses TF2 for map frame localization (works with AMCL)
- Hybrid obstacle validation (tight spaces + safety margins)
- Pure pursuit waypoint following
- Supports repeated experiments without restart

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
Note: 
For nav2, make sure to set the Reliability Policy under the Local Costmap Topic to Best Effort. 
For setting goals for d_star_nav, make sure to enable the tool properties, and update the topic (goal_pose) for 2d goal pose to {namespace}/d_star_goal_pose.
To visualize the dynamic grid, make sure to add the topic after running the navigator under the topic dynamic_grid.

**Features:**
- Incremental pathfinding with dynamic replanning
- SLAM integration for obstacle discovery
- Lidar-based dynamic obstacle detection
- Automatic replanning when obstacles detected
- Real-time path visualization

---
### Jump Point Search (JPS) Navigator
Autonomous navigation using Jump Point Search algorithm with optimized pathfinding through uniform-cost grids.


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


### DWA Navigator

Reactive local navigation using Dynamic Window Approach with TF2/AMCL localization.
These instructions assume robot namespace `/don`. Replace with your robot's namespace if different.

#### Quick Start (with Localization)

**Terminal 1 - Localization:**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
ros2 daemon stop; ros2 daemon start

ros2 launch turtlebot4_navigation localization.launch.py \
    namespace:=/don \
    map:=/path/to/your/map.yaml \
    params_file:=/path/to/obstacle-avoidance-comps/ros2_ws/amcl_params.yaml
```

**Terminal 2 - RViz:**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash

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


**Terminal 3 - DWA Node:**
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash
cd ~/obstacle-avoidance-comps/ros2_ws
colcon build
source install/setup.bash

ros2 run dwa_package dwa_node
```

Then in RViz:
1. Set initial pose with "2D Pose Estimate"
2. Send goals with "2D Goal Pose"

#### DWA Parameters

**Cost Weights** (normalized 0-1):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights.goal` | 0.5 | Reward for progress toward goal |
| `weights.heading` | 0.05 | Reward for pointing at goal |
| `weights.velocity` | 0.2 | Reward for higher speeds |
| `weights.smoothness` | 0.05 | Reward for less angular velocity |
| `weights.obstacle` | 0.1 | Incentivizes being further from objects |

**Safety Distances** (meters):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `critical_radius` | 0.18 | Hard rejection boundary |
| `safe_distance` | 0.60 | Distance for max obstacle score |
| `emergency_stop_distance` | 0.17 | Triggers immediate stop |

**Example:** Run with custom weights:
```bash
ros2 run dwa_package dwa_node --ros-args \
    -p weights.goal:=0.4 \
    -p weights.obstacle:=0.5 \
    -p max_velocity:=0.5
```

---

## Project Structure

```
ros2_ws/
├── src/
│   ├── a_star/          # A* pathfinding implementation
│   ├── d_star/          # D* pathfinding implementation
│   ├── dwa_package/     # DWA local planner with TF2 localization
│   ├── demo_package/    # Demo movement package
│   ├── joy_test/        # Position data testing
│   └── sensor_data/     # Camera and lidar visualization
├── maze_slamed.pgm      # Saved maze map (image)
└── maze_slamed.yaml     # Saved maze map (metadata)
```

---

## Troubleshooting

### TF2 transform not available
- With namespaced robots, TF is on `/don/tf` not `/tf`. Verify with: `ros2 topic hz /don/tf`
- Use `view_robot.launch.py` which handles TF remapping automatically
- To check TF chain manually: `ros2 run tf2_ros tf2_echo map base_link --ros-args -r /tf:=/don/tf -r /tf_static:=/don/tf_static`
- Verify map→odom→base_link chain exists

### Map frame not available in RViz
- The `map` frame is created by AMCL. It won't exist until AMCL initializes with a pose
- Use `amcl_params.yaml` with `set_initial_pose: true` to auto-bootstrap the map frame on startup
- If not using the params file, publish an initial pose: `ros2 topic pub -r 1 /don/initialpose ...` (use `-r 1` not `--once` — the single-shot publish often fires before AMCL discovers it)

### Map shows "no map received" in RViz
- Verify the map is publishing: `ros2 topic info /don/map` (should show 1 publisher)
- In RViz, expand the Map display properties and set **Durability** to **Transient Local**
- The map_server uses transient local QoS; RViz defaults to volatile which won't receive it

### Robot not localizing properly
- Set accurate initial pose using **2D Pose Estimate** in RViz
- Ensure the 2D Pose Estimate topic is set to `/don/initialpose` (Panels → Tool Properties)
- Ensure the map matches the actual environment
- Check particle cloud convergence (should cluster around robot)
- Try moving robot slightly to help AMCL converge

### Goals not being received
- Check topic name matches: `ros2 topic echo /don/goal_pose`
- Verify "2D Goal Pose" tool in RViz is configured to correct topic
- Panels → Tool Properties → 2D Goal Pose → Topic: `/don/goal_pose`

### AMCL stuck on "Waiting for service map_server/get_state"
- Check that `RMW_IMPLEMENTATION` is not set to an uninstalled DDS: `echo $RMW_IMPLEMENTATION`
- If set to `rmw_cyclonedds_cpp` without CycloneDDS installed, fix with: `export RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- Nodes may be crashing silently — check logs: `ls ~/.ros/log/` and inspect the latest launch.log

### Commands not found
- Source the workspace: `source install/setup.bash`
- Verify ROS 2 is sourced: `echo $ROS_DISTRO`