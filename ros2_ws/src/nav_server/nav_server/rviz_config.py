#!/usr/bin/env python3
"""Build a temp .rviz config tailored to the active planner combination.

Called at launch time by the unified launch file; the resulting file path
is passed to rviz2 via ``-d``.
"""

import os
import tempfile


def _grid():
    return """    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true"""


def _robot_model():
    return """    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Mass Properties:
        Inertia: false
        Mass: false
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true"""


def _tf():
    return """    - Class: rviz_default_plugins/TF
      Enabled: false
      Filter (blacklist): ""
      Filter (whitelist): ""
      Frame Timeout: 15
      Frames:
        All Enabled: false
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: false"""


def _laser_scan(ns):
    return f"""    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 47
      Min Color: 0; 0; 0
      Min Intensity: 47
      Name: LaserScan
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Points
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: {ns}/scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true"""


def _map(ns):
    return f"""    - Alpha: 0.699999988079071
      Binary representation: false
      Binary threshold: 100
      Class: rviz_default_plugins/Map
      Color Scheme: map
      Draw Behind: false
      Enabled: true
      Name: Map
      Topic:
        Depth: 5
        Durability Policy: Transient Local
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/map
      Update Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/map_updates
      Use Timestamp: false
      Value: true"""


def _path(ns, topic_suffix, name, color='25; 255; 0'):
    return f"""    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: {color}
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Billboards
      Line Width: 0.029999999329447746
      Name: {name}
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 255; 85; 255
      Pose Style: None
      Radius: 0.029999999329447746
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/{topic_suffix}
      Value: true"""


def _marker(ns, topic_suffix, name):
    return f"""    - Class: rviz_default_plugins/Marker
      Enabled: true
      Name: {name}
      Namespaces:
        {{}}
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/{topic_suffix}
      Value: true"""


def _occupancy_grid(ns, topic_suffix, name):
    return f"""    - Alpha: 0.5
      Binary representation: false
      Binary threshold: 100
      Class: rviz_default_plugins/Map
      Color Scheme: costmap
      Draw Behind: true
      Enabled: true
      Name: {name}
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/{topic_suffix}
      Update Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: ""
      Use Timestamp: false
      Value: true"""


def _dwa_displays(ns):
    return [
        _marker(ns, 'debug_obstacles', 'DWA Obstacles'),
        _marker(ns, 'dwa/trajectories', 'DWA Trajectories'),
        _marker(ns, 'dwa/best_trajectory', 'DWA Best Trajectory'),
    ]


def _astar_displays(ns):
    return [
        _path(ns, 'a_star/plan', 'A* Global Path', '25; 255; 0'),
    ]


def _dstar_displays(ns):
    return [
        _path(ns, 'd_star/plan', 'D* Path', '255; 165; 0'),
        _occupancy_grid(ns, 'dynamic_grid', 'D* Dynamic Grid'),
    ]


def _jps_displays(ns):
    return [
        _path(ns, 'jps/plan', 'JPS Path', '0; 200; 255'),
    ]


def generate_rviz_config(ns, global_planner, local_planner):
    """Return a complete .rviz YAML string for the given planner combination."""
    displays = [
        _grid(),
        _robot_model(),
        _tf(),
        _laser_scan(ns),
        _map(ns),
    ]

    planner_map = {
        'a_star': _astar_displays,
        'd_star': _dstar_displays,
        'jps': _jps_displays,
    }
    if global_planner in planner_map:
        displays.extend(planner_map[global_planner](ns))

    if local_planner == 'dwa' or global_planner == 'd_star':
        displays.append(
            _occupancy_grid(ns, 'obstacle_grid', 'LIDAR Obstacle Grid'))

    if local_planner == 'dwa':
        displays.extend(_dwa_displays(ns))

    displays_block = '\n'.join(displays)

    config = f"""Panels:
  - Class: rviz_common/Displays
    Help Height: 0
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
      Splitter Ratio: 0.5833333134651184
    Tree Height: 973
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /2D Pose Estimate1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
{displays_block}
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: clicked_point
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/goal_pose
    - Class: rviz_default_plugins/SetInitialPose
      Covariance x: 0.25
      Covariance y: 0.25
      Covariance yaw: 0.06853891909122467
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: {ns}/initialpose
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10.0
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 1.5247963666915894
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz_default_plugins)
      Yaw: 3.14
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1080
  Hide Left Dock: false
  Hide Right Dock: true
  Selection:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: true
  Width: 1920
  X: 0
  Y: 0
"""
    return config


def write_rviz_config(ns, global_planner, local_planner, output_dir=None):
    """Write the generated config to ``output_dir`` (defaults to system temp)."""
    content = generate_rviz_config(ns, global_planner, local_planner)

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    gp = global_planner if global_planner != 'none' else 'noglobal'
    lp = local_planner if local_planner != 'none' else 'nolocal'
    ns_clean = ns.strip('/').replace('/', '_')
    filename = f'nav_{ns_clean}_{gp}_{lp}.rviz'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(content)

    return filepath
