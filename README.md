# Obstacle Avoidance

ROS 2 workspace for TurtleBot 4 obstacle avoidance. One launch file runs any combination of
global planner (A\*, D\* Lite, JPS) and local planner (DWA) on top of a shared LIDAR
occupancy grid, with RViz.

Works on Humble and Jazzy. Check yours with `echo $ROS_DISTRO`.

Authors and project history: [AUTHORS.md](AUTHORS.md).

---

## Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Launch Options](#launch-options)
4. [Structure](#structure)
5. [Sending Goals](#sending-goals)
6. [Package Reference and Parameters](#package-reference-and-parameters)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Build

```bash
cd ~/obstacle-avoidance/ros2_ws
colcon build
source install/setup.bash
```

Interfaces must be built before the packages that use them, so incremental builds go:

```bash
colcon build --packages-select nav_interfaces
source install/setup.bash
colcon build --packages-select nav_server a_star d_star jps dwa_package obstacle_grid
source install/setup.bash
```

### 2. Install deps

```bash
sudo apt install ros-${ROS_DISTRO}-nav2-map-server ros-${ROS_DISTRO}-nav2-amcl
```

### 3. Shell setup

In `~/.bashrc`:

```bash
source /opt/ros/jazzy/setup.bash                 # or humble
source /etc/turtlebot4_discovery/setup.bash      # physical robot
ros2 daemon stop; ros2 daemon start
```

### 4. Save a map (first run only)

Drive under SLAM, then save:

```bash
# Terminal 1 — SLAM
ros2 launch turtlebot4_navigation slam.launch.py namespace:=/don

# Terminal 2 — teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -p stamped:=true -r /cmd_vel:=/don/cmd_vel

# Terminal 3 — save
ros2 run nav2_map_server map_saver_cli -f my_map \
    --ros-args -p map_subscribe_transient_local:=true -r __ns:=/don \
    -p save_map_timeout:=5000.0
```

You get `my_map.yaml` and `my_map.pgm`.

---

## Quick Start

Two terminals. Undock first — the LIDAR is off while docked and AMCL needs it.

**Terminal 1 — AMCL:**

```bash
ros2 launch turtlebot4_navigation localization.launch.py \
    namespace:=/don \
    map:=/path/to/my_map.yaml \
    params_file:=/path/to/obstacle-avoidance/ros2_ws/amcl_params.yaml
```

Wait for `Setting pose`. The bundled `amcl_params.yaml` seeds a pose at (0,0,0) so the
`map` frame exists immediately; refine it later with 2D Pose Estimate.

**Terminal 2 — planners, action server, RViz:**

```bash
cd ~/obstacle-avoidance/ros2_ws
source install/setup.bash

# A* + DWA (default)
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml
```

In RViz, click 2D Pose Estimate to refine the pose, then 2D Goal Pose to navigate.

---

## Launch Options

```bash
# A* + DWA stacked
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml

# D* Lite + DWA
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml global_planner:=d_star

# JPS + DWA
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml global_planner:=jps

# A* standalone (A* drives)
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml local_planner:=none

# D* standalone
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml \
    global_planner:=d_star local_planner:=none

# JPS standalone
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml \
    global_planner:=jps local_planner:=none

# DWA alone (no map, reactive only)
ros2 launch nav_server navigation_launch.py global_planner:=none local_planner:=dwa

# Different namespace
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml namespace:=/robot

# No RViz
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml rviz:=false

# No action server
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml server:=false
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `map_yaml` | *(empty)* | Map YAML path. Required for A\*, D\*, JPS |
| `namespace` | `/don` | Robot namespace; prefixed onto every topic |
| `global_planner` | `a_star` | `a_star`, `d_star`, `jps`, or `none` |
| `local_planner` | `dwa` | `dwa` or `none` |
| `rviz` | `true` | Launch RViz |
| `server` | `true` | Launch the action server |
| `params_file` | `stacked_params.yaml` | Shared parameter YAML |

---

## Structure

```
                     map_yaml ──────────────┐
                                            ▼
/{ns}/goal_pose  ─►  Global planner  ──►  /{ns}/{planner}/plan
(RViz or CLI)        (A*, D*, JPS)                │
                          ▲                       ▼
                          │                   Local planner        ─►  /{ns}/cmd_vel
                          │                   (DWA)
                          │                       │
                  /{ns}/obstacle_grid             │
                          ▲                       │
                          │                       │
                      obstacle_grid  ◄── /{ns}/scan (LIDAR)
                          ▲
                          │
                      /{ns}/map  (one-shot: dimensions + static walls)
```

`nav_server` subscribes to each planner's status topic and prints progress.

### Replanning

`obstacle_grid` rebuilds from every LIDAR scan, layered on top of the static map. Global
planners poll it so the path stays current:

| Planner | Replan |
|---------|--------|
| A\* | Timer, every `replan_interval` seconds (default 5 s) |
| D\* Lite | Incremental, on obstacle grid change |
| JPS | One-shot per goal |

DWA runs a 10 Hz dynamic-window search against the latest grid regardless.

### Stacked mode

When both `global_planner` and `local_planner` are set, the launch file passes
`stacked:=true` to both:

- The global planner publishes `nav_msgs/Path` on `/{ns}/{planner}/plan` and does not
  drive.
- DWA follows the path and drives `cmd_vel`, running its own state machine
  (`IDLE → NAVIGATING → EMERGENCY_STOPPED → RECOVERING`).

If only one is set, it runs standalone and drives directly.

### RViz config

Generated at launch time to match the selected planners:

- Always: Grid, RobotModel, TF, LaserScan (`/{ns}/scan`), Map (`/{ns}/map`), 2D Goal Pose
  (`/{ns}/goal_pose`), 2D Pose Estimate (`/{ns}/initialpose`)
- A\*: `/{ns}/a_star/plan` (green)
- D\*: `/{ns}/d_star/plan` (orange), `/{ns}/dynamic_grid` overlay
- JPS: `/{ns}/jps/plan` (blue)
- DWA: `/{ns}/debug_obstacles`, `/{ns}/dwa/trajectories`, `/{ns}/dwa/best_trajectory`

TF is remapped `/tf` → `/{ns}/tf` automatically.

---

## Sending Goals

### RViz

1. 2D Pose Estimate on the robot's actual location (once, or after you move it).
2. 2D Goal Pose on the target.
3. The launch terminal prints progress:
   ```
   [INFO] RViz goal received: (1.50, 2.00, 0.79)
   [INFO]   [NAVIGATING          ] to_goal: 1.23m  traveled: 0.45m  time: 3.2s  pos: (0.72, 0.88)
   [INFO]
   [INFO]   == REACHED ==
   [INFO]   Distance traveled: 1.87 m
   [INFO]   Total time:        12.34 s
   [INFO]   Average speed:     0.152 m/s
   ```
4. A new goal at any time cancels the current one.
5. Default timeout is 5 min.

### CLI

```bash
ros2 run nav_server navigate -- --goal 1.5 2.0
ros2 run nav_server navigate -- --goal 1.5 2.0 0.79   # heading in rad
ros2 run nav_server navigate -- --cancel
```

### `Navigate` action

Served at `/{ns}/navigate`. No Nav2 dependency:

```
# Goal
string global_planner       # a_star, d_star, jps
string local_planner        # dwa, none
float64 goal_x
float64 goal_y
float64 goal_theta
---
# Result
bool success
float64 total_distance
float64 total_time
string final_state          # reached, cancelled, timeout, failed
---
# Feedback
float64 distance_to_goal
float64 distance_traveled
float64 elapsed_time
string nav_state            # planning, navigating, emergency_stopped, recovering
float64 current_x
float64 current_y
```

Preemption is a `CancelNav` service call on each planner.

---

## Package Reference and Parameters

Override any param with `-p name:=value` on `ros2 run`, or via `params_file:=` on the
launch file.

### `nav_server`

Launch file, action server, CLI, RViz config generator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `global_planner` | `a_star` | Which planner to monitor/cancel (set by launch) |
| `local_planner` | `dwa` | Which local planner to monitor/cancel (set by launch) |
| `navigation_timeout` | `300.0` | Max nav time in seconds; 0 disables |

### `a_star`

Periodic replanning with RDP simplification and cubic-spline smoothing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `stacked` | `false` | Planner-only mode |
| `robot_radius` | `0.25` | Robot radius (m) |
| `safety_clearance` | `0.05` | Extra inflation margin (m) |
| `linear_speed` | `0.2` | Standalone forward speed (m/s) |
| `angular_speed` | `0.5` | Standalone rotation speed (rad/s) |
| `position_tolerance` | `0.1` | Waypoint reached threshold (m) |
| `angle_tolerance` | `0.1` | Angular alignment threshold (rad) |
| `max_path_waypoints` | `20` | Max waypoints after simplification (50 in stacked) |
| `tight_space_radius` | `5` | Cells near start that skip inflation |
| `max_search_nodes` | `100000` | A\* search cap |
| `simplification_epsilon` | `0.1` | RDP tolerance (m) |
| `replan_interval` | `5.0` | Replan period (s); 0 disables |

### `d_star`

D\* Lite. Replans incrementally when the obstacle grid changes.

```bash
ros2 run d_star d_star_nav --ros-args -p namespace:=/don
ros2 run d_star live_visualizer --ros-args -p namespace:=/don   # planned vs traveled
```

Publishes `/{ns}/d_star/plan` and `/{ns}/dynamic_grid`. Params: `namespace`, `stacked`,
`robot_radius`, `safety_clearance`.

### `jps`

Jump Point Search. One-shot per goal, fastest on large uniform grids.

```bash
ros2 run jps jps_nav --ros-args -p namespace:=/don
```

Publishes `/{ns}/jps/plan`. Params: `namespace`, `stacked`, `robot_radius`,
`safety_clearance`.

### `dwa_package`

Dynamic Window Approach. Needs `obstacle_grid`, which the launch file starts for you when
DWA is selected.

**Cost weights** (7 terms, normalized 0–1):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights.goal` | 0.35 | Progress toward goal |
| `weights.heading` | 0.05 | Pointing at goal (skipped in stacked) |
| `weights.velocity` | 0.10 | Prefer higher speed |
| `weights.smoothness` | 0.05 | Prefer lower angular velocity |
| `weights.obstacle` | 0.40 | Distance from obstacles |
| `weights.dist_path` | 0.10 | Stay near global path (stacked only) |
| `weights.heading_path` | 0.05 | Align with path tangent (stacked only; blends to 0.25 near goal) |

**Velocity and sampling:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_velocity` | 0.4 | Forward cap (m/s) |
| `min_velocity` | 0.0 | Forward min (m/s) |
| `max_angular_velocity` | 1.8 | Rotation cap (rad/s) |
| `min_angular_velocity` | -1.8 | Rotation min (rad/s) |
| `max_linear_acceleration` | 0.5 | Linear accel (m/s²) |
| `max_angular_acceleration` | 2.0 | Angular accel (rad/s²) |
| `v_samples` | 20 | Linear samples |
| `w_samples` | 20 | Angular samples |
| `dt` | 0.1 | Sim step (s) |
| `prediction_steps` | 25 | Trajectory lookahead (× dt) |
| `window_steps` | 5 | Dynamic window size (× dt) |
| `grid_size` | 161 | Local window side (cells) |
| `grid_resolution` | 10.0 | Local window (cells/m) |

**Safety and recovery:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `critical_radius` | 0.20 | Hard rejection distance (m) |
| `emergency_stop_distance` | 0.17 | Immediate stop on LIDAR point (m) |
| `max_lidar_range` | 8.0 | Ignore returns past this (m) |
| `goal_tolerance` | 0.2 | Goal-reached threshold (m) |
| `lookahead` | 0.85 | Stacked-mode carrot distance (m) |
| `max_path_deviation` | 1.0 | Path distance that scores 0 for `dist_path` (m) |
| `recovery.angular_velocity` | 0.5 | Rotate-in-place speed (rad/s) |
| `recovery.backup_velocity` | -0.05 | Reverse speed (m/s) |
| `recovery.rotate_timeout` | 3.0 | Rotate time before backup (s) |
| `recovery.total_timeout` | 10.0 | Max recovery before abort (s) |

### `obstacle_grid`

Shared LIDAR occupancy grid in the map frame. Bresenham raycasting clears cells the
moment a ray passes through; cells that haven't been reconfirmed in `obstacle_decay_seconds`
fall back to free. Needs AMCL (for `map ← odom` TF).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obstacle_decay_seconds` | 20.0 | Fallback decay (s) |
| `robot_radius` | 0.22 | Inflation radius (m) |
| `safety_clearance` | 0.05 | Extra inflation (m) |
| `publish_rate` | 10.0 | Hz |
| `max_lidar_range` | 8.0 | m |
| `lidar_downsample` | 2 | Every Nth beam |

Publishes `/{ns}/obstacle_grid`. Subscribes `/{ns}/scan` continuously, `/{ns}/map` once
for dimensions.

### `nav_interfaces`

CMake package with the action/msg/srv definitions:

- `action/Navigate.action` — goal/result/feedback shown above
- `msg/NavStatus.msg` — globals publish on `/{ns}/{planner}/status`, DWA on
  `/{ns}/dwa/nav_status`
- `srv/CancelNav.srv` — preemption, one per planner

---

## Troubleshooting

### AMCL lifecycle hangs

Symptom: `failed to send response to /don/map_server/change_state (timeout)`.

```bash
# Second terminal. Each may hang for a few seconds — Ctrl+C and keep going.
ros2 lifecycle set /don/map_server activate
ros2 lifecycle set /don/amcl configure
ros2 lifecycle set /don/amcl activate

# Seed a pose so the map frame appears
ros2 topic pub -r 1 /don/initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

Use `-r 1`, not `--once`. The one-shot publish often fires before AMCL discovers it.

### Map shows "no map received"

`map_server` uses transient-local QoS; RViz defaults to volatile. In the Map display's
properties, set Durability to Transient Local.

### TF missing on a namespaced robot

TF is on `/don/tf`, not `/tf`. The launch file remaps automatically. Manual check:

```bash
ros2 run tf2_ros tf2_echo map base_link \
    --ros-args -r /tf:=/don/tf -r /tf_static:=/don/tf_static
```

### AMCL stuck on "Waiting for service map_server/get_state"

```bash
echo $RMW_IMPLEMENTATION
# If rmw_cyclonedds_cpp but CycloneDDS isn't installed:
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### Commands not found

```bash
source install/setup.bash
echo $ROS_DISTRO
```

### Robot doesn't move

- Undock it. LIDAR is off while docked.
- `ros2 topic echo /don/goal_pose` to confirm the goal arrived.
- In RViz, Panels → Tool Properties → 2D Goal Pose → topic is `/don/goal_pose`.
