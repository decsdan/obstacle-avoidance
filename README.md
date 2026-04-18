# Obstacle Avoidance

ROS 2 workspace for TurtleBot 4 obstacle avoidance. One launch file brings up a global
planner (A\*, D\* Lite, JPS), a local planner (DWA), a shared LIDAR obstacle grid, and
the navigation action server that ties them together, with RViz.

Works on Humble and Jazzy. Check yours with `echo $ROS_DISTRO`.

Authors and project history: [AUTHORS.md](AUTHORS.md).

---

## Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Launch Options](#launch-options)
4. [Architecture](#architecture)
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
# A* + DWA (default)
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml

# D* Lite + DWA
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml global_planner:=d_star

# JPS + DWA
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml global_planner:=jps

# DWA alone (no global plan, reactive only)
ros2 launch nav_server navigation_launch.py global_planner:=none local_planner:=dwa

# Different namespace
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml namespace:=/robot

# No RViz
ros2 launch nav_server navigation_launch.py map_yaml:=/path/to/my_map.yaml rviz:=false

# No action server (launch just the planners/grid)
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
| `server` | `true` | Launch the navigation action server |

Each planner ships its own params YAML in its package's `config/` directory; the launch
file loads the one that matches the selected planner.

---

## Architecture

The stack is a strict call/response orchestrator. `nav_server` owns the `Navigate`
action and drives three contracts:

```
                     Navigate goal (RViz / CLI)
                              │
                              ▼
                         nav_server
                   ┌──────────┼─────────────┐
                   │          │             │
                   ▼          ▼             ▼
        GetGridSnapshot   PlanPath     FollowPath (action)
         (obstacle_grid)  (a_star |     (dwa)
                           d_star |        │
                           jps)            ▼
                                      /{ns}/cmd_vel
```

No planner drives the robot directly. No planner runs its own replan timer. The action
server makes all scheduling decisions: when to request a new path, when to cancel the
current follow, when to abort on a safety latch.

### Contracts

- **`GetGridSnapshot` service** (`{ns}/get_grid_snapshot`): returns a matched `(raw,
  inflated, stamp)` triple so planners see a consistent grid without racing topic
  subscriptions.
- **`PlanPath` service** (`{ns}/{planner}/plan_path`): takes a grid snapshot plus
  start/goal, returns a path plus diagnostics (`nodes_expanded`, `cached`,
  `failure_reason`).
- **`FollowPath` action** (`{ns}/{local}/follow_path`): takes a reference path,
  streams feedback (pose, velocity, distance_to_goal, cross_track_error, local_state),
  terminates with `reached` | `path_blocked` | `cancelled` | `failed`.
- **`Navigate` action** (`{ns}/navigate`): the top-level contract. Goal is a
  `PoseStamped` plus `global_planner`, `local_planner`, `global_budget`, `scenario_id`.

### Replan triggers

`nav_server` decides when to replan; planners themselves never loop.

| Trigger | Source |
|---------|--------|
| `path_blocked` from DWA | FollowPath result |
| Pose deviates > `replan_deviation_threshold` from active path | Analyze-phase check |
| Periodic replan | `replan_period_sec` > 0 (off by default for determinism) |

D\* Lite keeps its search state alive across calls — when `nav_server` calls it again
with the same goal, the planner reuses the previous computation and only touches
vertices where the grid changed. A\* and JPS run fresh each call.

### RViz config

Generated at launch time to match the selected planners:

- Always: Grid, RobotModel, TF, LaserScan (`{ns}/scan`), Map (`{ns}/map`), 2D Goal
  Pose (`{ns}/goal_pose`), 2D Pose Estimate (`{ns}/initialpose`)
- Any global planner: Active Path (`{ns}/nav_server/active_path`), colored per planner
- DWA: LIDAR Obstacle Grid (`{ns}/obstacle_grid`) and DWA marker topics
- D\*: adds the obstacle grid display even without DWA

TF is remapped `/tf` → `{ns}/tf` automatically.

---

## Sending Goals

### RViz

1. 2D Pose Estimate on the robot's actual location (once, or after you move it).
2. 2D Goal Pose on the target.
3. The launch terminal prints progress:
   ```
   [INFO] RViz goal → (1.50, 2.00)
   [INFO] [execute] scenario=rviz global=a_star local=dwa goal=(1.50, 2.00)
   [INFO] [plan] path received: 18 poses in 0.042s (nodes_expanded=312, cached=False)
   [INFO] [execute] FollowPath terminal=reached distance=1.87 time=12.34
   [INFO] [result] outcome=reached distance=1.87 time=12.34 replans=0
   ```
4. A new 2D Goal Pose at any time cancels the current goal.
5. Default timeout is 5 min (`navigation_timeout`).

### CLI

```bash
ros2 run nav_server navigate -- --goal 1.5 2.0
ros2 run nav_server navigate -- --goal 1.5 2.0 0.79        # heading in rad
ros2 run nav_server navigate -- --goal 1.5 2.0 --global-planner d_star
ros2 run nav_server navigate -- --goal 1.5 2.0 --scenario-id test_01
```

Ctrl+C cancels the active goal.

### `Navigate` action

Served at `{ns}/navigate`. The full schema lives in
[`nav_interfaces/action/Navigate.action`](ros2_ws/src/nav_interfaces/action/Navigate.action):

```
# Goal
geometry_msgs/PoseStamped goal
string global_planner        # "a_star" | "d_star" | "jps"
string local_planner         # "dwa"
float64 global_budget        # PlanPath budget; 0 = planner default
string scenario_id           # Optional; correlates the run with a scenario record
---
# Result
bool success
string terminal_outcome      # "reached" | "path_blocked" | "timeout" | "cancelled" | "failed"
float64 total_distance
float64 total_time
uint32 replan_count
---
# Feedback (published at feedback_hz)
float64 distance_to_goal
float64 distance_traveled
float64 elapsed_time
string nav_state             # "planning" | "following" | "replanning" | "recovering"
geometry_msgs/Pose current_pose
geometry_msgs/Twist current_velocity
```

Cancellation is standard ROS 2 action cancellation; nav_server propagates it to the
active FollowPath goal.

---

## Package Reference and Parameters

Each planner ships its own YAML in `config/`. Override any param with `-p name:=value`
on `ros2 run`, or pass a replacement file with `--ros-args --params-file`.

### `nav_server`

Action server, CLI, RViz config generator, launch file.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `map_frame` | `map` | TF frame for the map |
| `base_frame` | `base_link` | TF frame for the robot |
| `default_global_planner` | `a_star` | Fallback when Navigate goal leaves it blank |
| `default_local_planner` | `dwa` | Fallback when Navigate goal leaves it blank |
| `navigation_timeout` | `300.0` | Max Navigate duration (s); 0 disables |
| `plan_timeout` | `5.0` | PlanPath `wait_for_service` timeout (s) |
| `feedback_hz` | `10.0` | Navigate feedback rate |
| `replan_on_path_blocked` | `true` | Replan when DWA reports path_blocked |
| `replan_deviation_threshold` | `0.5` | Meters from nearest path point before replan |
| `replan_period_sec` | `0.0` | Periodic replan interval (s); 0 = off |

Config file: `nav_server/config/nav_server_params.yaml`.

### `a_star`

A\* with RDP simplification and cubic-spline smoothing. Service-only.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `max_search_nodes` | `100000` | A\* expansion cap |
| `simplification_epsilon` | `0.1` | RDP tolerance (m) |
| `max_path_waypoints` | `50` | Max waypoints after simplification |
| `tight_space_radius` | `5` | Cells near start that skip inflation |

Serves `{ns}/a_star/plan_path`. Config: `a_star/config/a_star_params.yaml`.

### `d_star`

D\* Lite with persistent search state. On repeat calls with the same goal and grid
frame, only vertices near changed cells get re-touched; the response's `cached` flag
reports reuse.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `max_waypoints` | `50` | Max waypoints in returned path |
| `tight_space_radius` | `5` | Cells near start that skip inflation |
| `max_iterations` | `10000000` | compute_shortest_path expansion cap |

Serves `{ns}/d_star/plan_path`. Config: `d_star/config/d_star_params.yaml`.

### `jps`

Jump Point Search. One-shot per call, fastest on large uniform grids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `max_path_waypoints` | `50` | Max waypoints after simplification |
| `tight_space_radius` | `5` | Cells near start that skip inflation |
| `simplification_epsilon` | `0.1` | RDP tolerance (m) |
| `max_jump_depth` | `500` | Per-direction jump recursion cap |
| `max_search_nodes` | `100000` | Open-list expansion cap |

Serves `{ns}/jps/plan_path`. Config: `jps/config/jps_params.yaml`.

### `dwa_package`

Dynamic Window Approach as a FollowPath action server. Needs `obstacle_grid`, which
the launch file starts for you when DWA is selected.

**Cost weights** (normalized 0–1):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights.goal` | 0.35 | Progress toward goal |
| `weights.velocity` | 0.10 | Prefer higher speed |
| `weights.smoothness` | 0.05 | Prefer lower angular velocity |
| `weights.obstacle` | 0.40 | Distance from obstacles |
| `weights.dist_path` | 0.10 | Stay near reference path |
| `weights.heading_path` | 0.05 | Align with path tangent |

**Velocity and sampling:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_velocity` | 0.4 | Forward cap (m/s) |
| `max_angular_velocity` | 1.8 | Rotation cap (rad/s) |
| `max_linear_acceleration` | 0.5 | Linear accel (m/s²) |
| `max_angular_acceleration` | 2.0 | Angular accel (rad/s²) |
| `v_samples` | 20 | Linear samples |
| `w_samples` | 20 | Angular samples |
| `dt` | 0.1 | Sim step (s) |
| `prediction_steps` | 25 | Trajectory lookahead (× dt) |
| `window_steps` | 5 | Dynamic window size (× dt) |
| `grid_size` | 161 | Local window side (cells) |
| `grid_resolution` | 10.0 | Local window (cells/m) |

**Safety and follow semantics:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `critical_radius` | 0.20 | Hard rejection distance (m) |
| `emergency_stop_distance` | 0.17 | Immediate stop on LIDAR point (m) |
| `max_lidar_range` | 8.0 | Ignore returns past this (m) |
| `goal_tolerance` | 0.2 | Goal-reached threshold (m) |
| `lookahead` | 0.85 | Carrot distance along reference path (m) |
| `max_path_deviation` | 1.0 | Path distance that scores 0 for `dist_path` (m) |
| `blocked_ticks_to_abort` | 10 | Consecutive rejected-trajectory ticks → path_blocked |
| `stuck_timeout` | 8.0 | E-stop hold time before giving up (s) |

Serves `{ns}/dwa/follow_path`. Config: `dwa_package/config/dwa_params.yaml`.

### `obstacle_grid`

Shared LIDAR occupancy grid in the map frame. Bresenham raycasting clears cells the
moment a ray passes through; cells that haven't been reconfirmed in
`obstacle_decay_seconds` fall back to free. Needs AMCL (for `map ← odom` TF).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `namespace` | `/don` | Robot namespace |
| `obstacle_decay_seconds` | 20.0 | Fallback decay (s) |
| `robot_radius` | 0.22 | Inflation radius (m) |
| `safety_clearance` | 0.05 | Extra inflation (m) |
| `publish_rate` | 10.0 | Update cycle rate (Hz) |
| `max_lidar_range` | 8.0 | Max raycast (m) |
| `lidar_downsample` | 2 | Use every Nth beam |

Serves `{ns}/get_grid_snapshot` (raw + inflated pair). Also publishes
`{ns}/obstacle_grid` (inflated) and `{ns}/obstacle_grid_raw` (pre-inflation).
Subscribes `{ns}/scan` continuously, `{ns}/map` once for dimensions.

### `nav_interfaces`

CMake package with the action/srv definitions:

- `action/Navigate.action` — orchestrator entry point
- `action/FollowPath.action` — local planner contract
- `action/RunBatch.action` — batch-scenario runner (unused so far)
- `srv/PlanPath.srv` — global planner contract
- `srv/GetGridSnapshot.srv` — obstacle grid snapshot contract

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
- `ros2 topic echo /don/goal_pose` to confirm an RViz goal arrived.
- Check `ros2 action list` shows `/don/navigate`, `/don/dwa/follow_path`.
- Check `ros2 service list` shows `/don/get_grid_snapshot` and `/don/{planner}/plan_path`.
- In RViz, Tool Properties → 2D Goal Pose → topic is `/don/goal_pose`.
