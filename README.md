
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

ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/don
```

Then in RViz: File → Open Config → `~/obstacle-avoidance-comps/don_viz.rviz`

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

**Save configuration:**
1. File → Save Config As → `~/obstacle-avoidance-comps/don_viz.rviz`


### Terminal 3 - Debugging (optional)
```bash
source /opt/ros/jazzy/setup.bash
source /etc/turtlebot4_discovery/setup.bash

ros2 topic hz /don/scan
ros2 topic echo /don/goal_pose
ros2 topic echo /don/cmd_vel
```

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