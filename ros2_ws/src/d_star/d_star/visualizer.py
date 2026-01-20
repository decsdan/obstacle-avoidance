#!/usr/bin/env python3

"""
Interactive A* Path Visualizer for TurtleBot4

This visualizer provides an interactive matplotlib-based interface for testing
pathfinding algorithms with dynamic obstacle placement and robot simulation.

Features:
    - Interactive start/goal selection via mouse clicks
    - Dynamic obstacle placement mode
    - A* pathfinding with exploration visualization
    - Robot movement simulation with automatic replanning
    - Obstacle detection lookahead (simulates lidar-like behavior)
    - Real-time replanning when obstacles block the path

Usage:
    Run directly: python3 visualizer.py
    With custom map: python3 visualizer.py <yaml_file> <pgm_file>
    With custom robot params: python3 visualizer.py <yaml> <pgm> <radius> <clearance>

Interactive Controls:
    - Click: Set start (1st click), goal (2nd click), reset (3rd+ click)
    - 'o': Toggle obstacle placement mode
    - 'r': Run/replan pathfinding
    - 'm': Start/stop robot movement simulation
    - 'a': Animate search process
    - 'c': Clear all and reset

Architecture:
    - Uses A* for pathfinding (not D* Lite, despite class name)
    - Maintains static map (inflated) and dynamic map (with placed obstacles)
    - Robot simulation checks lookahead distance for obstacles
    - Automatic replanning when obstacles detected in path
"""

import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import math


# ============================================================================
# CONSTANTS
# ============================================================================

class VisualizerConstants:
    """Configuration constants for the visualizer"""

    # Robot Physical Parameters (meters)
    ROBOT_RADIUS = 0.22          # TurtleBot4 radius
    SAFETY_CLEARANCE = 0.001      # Additional safety margin

    # Simulation Parameters
    LOOKAHEAD_DISTANCE = 1.0     # Distance to check for obstacles ahead (meters)
    MAX_PATH_HISTORY = 1000      # Maximum traveled path points to store

    # Animation Parameters
    ANIMATION_INTERVAL = 10      # Milliseconds between animation frames
    WAYPOINTS_PER_FRAME = 3      # Waypoints to move per animation frame

    # Obstacle Placement
    OBSTACLE_RADIUS_PIXELS = 5   # Radius of placed obstacles (grid cells)

    # Visualization
    FIGURE_SIZE = (12, 10)       # Figure dimensions (width, height)
    PATH_LINE_WIDTH = 3          # Width of path line
    ROBOT_MARKER_SIZE = 15       # Size of robot position marker
    GOAL_MARKER_SIZE = 20        # Size of goal marker

    # Colors (RGB)
    COLOR_FREE_SPACE = [1.0, 1.0, 1.0]      # White
    COLOR_CLEARANCE = [1.0, 0.7, 0.7]        # Light red
    COLOR_OBSTACLE = [0.2, 0.2, 0.2]         # Dark gray
    COLOR_DYNAMIC_OBS = 'orange'             # Orange for placed obstacles


# ============================================================================
# MAIN VISUALIZER CLASS
# ============================================================================

class DStarVisualizer:
    """
    Interactive pathfinding visualizer with dynamic obstacle placement.

    Note: Despite the name, this uses A* algorithm (not D* Lite).
    The name is kept for compatibility with the d_star package.
    """

    def __init__(self, robot_radius=None, safety_clearance=None):
        """
        Initialize the visualizer.

        Args:
            robot_radius: Robot radius in meters (default from constants)
            safety_clearance: Safety margin in meters (default from constants)
        """
        # ====================================================================
        # INITIALIZATION - Robot Parameters
        # ====================================================================

        self.robot_radius = robot_radius if robot_radius is not None else VisualizerConstants.ROBOT_RADIUS
        self.safety_clearance = safety_clearance if safety_clearance is not None else VisualizerConstants.SAFETY_CLEARANCE

        # ====================================================================
        # INITIALIZATION - Map Data
        # ====================================================================

        self.grid = None                # Inflated static map (occupied=1, free=0)
        self.grid_original = None       # Original map before inflation
        self.grid_dynamic = None        # Dynamic map with placed obstacles
        self.resolution = None          # Meters per grid cell
        self.origin = None              # Map origin [x, y, theta]

        # ====================================================================
        # INITIALIZATION - Planning State
        # ====================================================================

        self.start = None               # Start position (grid coordinates)
        self.goal = None                # Goal position (grid coordinates)
        self.path = []                  # Current planned path (grid coordinates)
        self.explored_nodes = []        # Nodes explored during search
        self.open_set_viz = []          # Open set history for visualization
        self.dynamic_obstacles = []     # User-placed obstacles (grid coordinates)

        # ====================================================================
        # INITIALIZATION - Interactive State
        # ====================================================================

        self.click_count = 0                    # Track clicks: 0=none, 1=start set, 2=goal set
        self.mode = 'navigate'                  # 'navigate' or 'add_obstacle'

        # ====================================================================
        # INITIALIZATION - Robot Simulation
        # ====================================================================

        self.robot_position = None              # Current robot position (grid coords)
        self.robot_path_index = 0               # Index in path for simulation
        self.animation_running = False          # Is robot simulation active?
        self.robot_animation = None             # FuncAnimation object for robot
        self.lookahead_distance = VisualizerConstants.LOOKAHEAD_DISTANCE

        # ====================================================================
        # INITIALIZATION - Visualization
        # ====================================================================

        self.fig = None                         # Matplotlib figure
        self.ax = None                          # Matplotlib axis

    # ========================================================================
    # MAP LOADING AND PROCESSING
    # ========================================================================

    def load_map(self, yaml_file, pgm_file):
        """
        Load map from YAML and PGM files.

        Loads the static map, applies coordinate transforms to match Gazebo,
        inflates obstacles by robot radius + safety clearance, and initializes
        the dynamic grid for obstacle placement.

        Args:
            yaml_file: Path to map metadata YAML file
            pgm_file: Path to map image PGM file
        """
        try:
            # Load map metadata
            with open(yaml_file, 'r') as f:
                map_data = yaml.safe_load(f)

            self.resolution = map_data['resolution']
            self.origin = map_data['origin']

            # Load map image
            img = Image.open(pgm_file)
            occupancy_grid = np.array(img)

            # Convert to binary: 0=free, 1=occupied
            # PGM: 255=free, <250=occupied/unknown
            self.grid = np.zeros_like(occupancy_grid)
            self.grid[occupancy_grid < 250] = 1  # Occupied/unknown
            self.grid[occupancy_grid >= 250] = 0  # Free space

            # Flip map vertically to match Gazebo coordinate system
            self.grid = np.flipud(self.grid)

            # Store original grid before inflation (for visualization)
            self.grid_original = self.grid.copy()

            # Inflate obstacles for robot safety
            total_inflation = self.robot_radius + self.safety_clearance
            self.inflate_obstacles(total_inflation)

            # Initialize dynamic grid (copy of inflated grid)
            self.grid_dynamic = self.grid.copy()

            # Print configuration
            print(f'Map loaded: {self.grid.shape}, resolution: {self.resolution}')
            print(f'Robot radius: {self.robot_radius}m, Safety clearance: {self.safety_clearance}m')
            print(f'Total obstacle inflation: {total_inflation}m ({int(total_inflation/self.resolution)} pixels)')
            print(f'\nD* Lite Visualizer - Interactive Features:')
            print(f'  - Click to set start and goal positions')
            print(f'  - Press "o" to toggle obstacle placement mode')
            print(f'  - Press "r" to run pathfinding')
            print(f'  - Press "m" to start robot simulation (auto-replans if obstacles detected)')
            print(f'  - Press "c" to clear and start over')
            print(f'  - Lookahead distance: {self.lookahead_distance}m (automatic obstacle detection)')
        except Exception as e:
            print(f'Failed to load map: {e}')

    def inflate_obstacles(self, inflation_radius):
        """
        Inflate obstacles by specified radius using morphological dilation.

        Creates a safety buffer around all obstacles equal to robot_radius +
        safety_clearance. This ensures the robot's center can safely follow
        paths without collision.

        Args:
            inflation_radius: Inflation distance in meters
        """
        from scipy.ndimage import binary_dilation

        # Convert radius to pixels
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))

        # Apply dilation to original grid
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    # ========================================================================
    # COORDINATE CONVERSION
    # ========================================================================

    def world_to_grid(self, x, y):
        """
        Convert world coordinates (meters) to grid indices (cells).

        Args:
            x: X coordinate in world frame (meters)
            y: Y coordinate in world frame (meters)

        Returns:
            tuple: (grid_x, grid_y) in grid cell coordinates
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices (cells) to world coordinates (meters).

        Args:
            grid_x: X index in grid coordinates
            grid_y: Y index in grid coordinates

        Returns:
            tuple: (x, y) in world frame (meters)
        """
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    # ========================================================================
    # GRID VALIDATION
    # ========================================================================

    def is_valid(self, grid_pos, use_dynamic=False):
        """
        Check if a grid position is valid and free of obstacles.

        Args:
            grid_pos: Tuple (grid_x, grid_y) to check
            use_dynamic: If True, check dynamic grid; else check static grid

        Returns:
            bool: True if position is valid and free, False otherwise
        """
        gx, gy = grid_pos
        grid_to_check = self.grid_dynamic if use_dynamic else self.grid
        rows, cols = grid_to_check.shape

        # Check bounds
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        # Check if free (0 = free, 1 = occupied)
        return grid_to_check[gy, gx] == 0  # Note: grid is [row, col] = [y, x]

    # ========================================================================
    # A* PATHFINDING ALGORITHM
    # ========================================================================

    def astar_with_visualization(self, start, goal, use_dynamic=False):
        """
        A* pathfinding algorithm with step-by-step visualization data.

        Finds the shortest path from start to goal using A* search with
        Euclidean distance heuristic. Supports 8-connected grid movement
        with diagonal cost adjustment.

        Args:
            start: Start position as (grid_x, grid_y)
            goal: Goal position as (grid_x, grid_y)
            use_dynamic: If True, use dynamic grid with obstacles; else static

        Returns:
            list: Path as list of (grid_x, grid_y) tuples, or empty list if no path
        """
        # Heuristic: Euclidean distance
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Select grid to use
        grid_to_use = self.grid_dynamic if use_dynamic else self.grid
        rows, cols = grid_to_use.shape

        # Initialize A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        # Reset visualization data
        self.explored_nodes = []
        self.open_set_viz = []
        nodes_explored = 0

        # A* main loop
        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

            # Store for visualization
            self.explored_nodes.append(current)
            self.open_set_viz.append([node[1] for node in open_set])

            # Check if reached goal
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print(f'A* explored {nodes_explored} nodes (using {"dynamic" if use_dynamic else "static"} grid)')
                return path[::-1]  # Reverse to get start->goal order

            # Explore 8-connected neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                gx, gy = neighbor

                # Check bounds
                if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
                    continue

                # Check obstacles
                if grid_to_use[gy, gx] != 0:
                    continue

                # Diagonal moves cost sqrt(2) ≈ 1.414
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        print(f'A* failed after exploring {nodes_explored} nodes')
        return []

    def run_astar(self):
        """
        Run A* algorithm and store results.

        Executes pathfinding from current start to goal, using dynamic grid
        if obstacles have been placed. Updates internal path state and prints
        path statistics.
        """
        if self.start is None or self.goal is None:
            print("Please set both start and goal positions by clicking on the map")
            return

        # Use dynamic grid if obstacles have been placed
        use_dynamic = len(self.dynamic_obstacles) > 0

        print(f"\nRunning A* from {self.start} to {self.goal}...")
        if use_dynamic:
            print("Using dynamic grid (with placed obstacles)")

        # Run pathfinding
        self.path = self.astar_with_visualization(self.start, self.goal, use_dynamic=use_dynamic)

        if self.path:
            print(f"Path found with {len(self.path)} waypoints!")

            # Calculate path length in meters
            path_world = [self.grid_to_world(gx, gy) for gx, gy in self.path]
            path_length = sum([
                math.sqrt(
                    (path_world[i][0] - path_world[i+1][0])**2 +
                    (path_world[i][1] - path_world[i+1][1])**2
                )
                for i in range(len(path_world) - 1)
            ])
            print(f"Path length: {path_length:.2f} meters")

            # Reset robot position for simulation
            self.robot_position = self.start
            self.robot_path_index = 0
        else:
            print("No path found!")

    # ========================================================================
    # OBSTACLE MANAGEMENT
    # ========================================================================

    def check_obstacles_in_lookahead(self):
        """
        Check if there are obstacles within lookahead distance along the path.

        Simulates a forward-looking sensor (like lidar) that checks for
        obstacles along the planned path up to lookahead_distance meters ahead.

        Returns:
            bool: True if obstacle detected in lookahead window, False otherwise
        """
        if not self.path or self.robot_path_index >= len(self.path):
            return False

        current_pos = self.robot_position if self.robot_position else self.start
        current_world = self.grid_to_world(current_pos[0], current_pos[1])

        # Check waypoints ahead within lookahead distance
        cumulative_distance = 0.0
        prev_world = current_world

        for i in range(self.robot_path_index, len(self.path)):
            waypoint = self.path[i]
            waypoint_world = self.grid_to_world(waypoint[0], waypoint[1])

            # Calculate distance from previous waypoint
            segment_dist = math.sqrt(
                (waypoint_world[0] - prev_world[0])**2 +
                (waypoint_world[1] - prev_world[1])**2
            )
            cumulative_distance += segment_dist

            # Stop checking beyond lookahead distance
            if cumulative_distance > self.lookahead_distance:
                break

            # Check if this waypoint has an obstacle
            gx, gy = waypoint
            if self.grid_dynamic[gy, gx] != 0:
                return True

            prev_world = waypoint_world

        return False

    # ========================================================================
    # ROBOT SIMULATION
    # ========================================================================

    def start_robot_simulation(self):
        """
        Start robot movement simulation along the planned path.

        Creates an animation that moves the robot along the path, checking
        for obstacles ahead and replanning if necessary.
        """
        if not self.path:
            return

        self.robot_position = self.start
        self.robot_path_index = 0
        self.animation_running = True

        # Create animation
        from matplotlib.animation import FuncAnimation
        self.robot_animation = FuncAnimation(
            self.fig,
            self.update_robot_position,
            frames=len(self.path),
            interval=VisualizerConstants.ANIMATION_INTERVAL,
            repeat=False,
            blit=False
        )
        plt.draw()

    def update_robot_position(self, frame):
        """
        Update robot position during simulation and handle replanning.

        Called by FuncAnimation for each animation frame. Checks for obstacles
        in the lookahead window and replans if necessary. Moves robot forward
        along path if no obstacles detected.

        Args:
            frame: Animation frame number (from FuncAnimation)
        """
        # Check if simulation complete
        if not self.path or self.robot_path_index >= len(self.path):
            if self.robot_animation:
                self.robot_animation.event_source.stop()
            self.animation_running = False
            print("\n✓ Robot reached goal!")
            return

        # Check for obstacles in lookahead window
        if self.robot_path_index < len(self.path):
            obstacle_in_window = self.check_obstacles_in_lookahead()

            if obstacle_in_window:
                # Obstacle detected - replan
                print(f"\n⚠ Obstacle detected within {self.lookahead_distance}m! Replanning...")

                # Stop current animation
                if self.robot_animation:
                    self.robot_animation.event_source.stop()
                    self.robot_animation = None

                # Replan from current position
                current_grid_pos = self.robot_position if self.robot_position else self.start
                self.start = current_grid_pos

                print(f"  Replanning from {current_grid_pos} to {self.goal}...")
                new_path = self.astar_with_visualization(current_grid_pos, self.goal, use_dynamic=True)

                if new_path:
                    # New path found - restart simulation
                    self.path = new_path
                    self.robot_path_index = 0
                    print(f"  ✓ New path found with {len(new_path)} waypoints")

                    # Restart animation with new path
                    from matplotlib.animation import FuncAnimation
                    self.robot_animation = FuncAnimation(
                        self.fig,
                        self.update_robot_position,
                        frames=len(self.path),
                        interval=VisualizerConstants.ANIMATION_INTERVAL,
                        repeat=False,
                        blit=False
                    )
                    self.visualize_static()
                else:
                    # No path - stop simulation
                    print("  ✗ No path found! Stopping robot.")
                    self.animation_running = False
                return

            # No obstacles ahead - continue moving
            # Move multiple waypoints per frame for faster movement
            waypoints_per_frame = VisualizerConstants.WAYPOINTS_PER_FRAME
            for _ in range(waypoints_per_frame):
                if self.robot_path_index < len(self.path):
                    next_pos = self.path[self.robot_path_index]
                    self.robot_position = next_pos
                    self.robot_path_index += 1
                else:
                    break

        # Redraw visualization
        self.visualize_static()

    # ========================================================================
    # MOUSE AND KEYBOARD EVENT HANDLERS
    # ========================================================================

    def on_click(self, event):
        """
        Handle mouse clicks for start/goal selection or obstacle placement.

        Behavior depends on current mode:
        - Navigate mode: First click=start, second=goal, third=reset
        - Obstacle mode: Each click places an obstacle

        Args:
            event: Matplotlib button_press_event
        """
        if event.inaxes != self.ax:
            return

        # Convert click coordinates to grid coordinates
        x_click = event.xdata
        y_click = event.ydata
        grid_pos = self.world_to_grid(x_click, y_click)
        gx, gy = grid_pos

        # Check bounds
        rows, cols = self.grid.shape
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            print(f"Click out of bounds: {grid_pos}")
            return

        # --------------------------------------------------------------------
        # OBSTACLE PLACEMENT MODE
        # --------------------------------------------------------------------
        if self.mode == 'add_obstacle':
            # Add circular obstacle to dynamic grid
            obstacle_radius = VisualizerConstants.OBSTACLE_RADIUS_PIXELS
            for dx in range(-obstacle_radius, obstacle_radius + 1):
                for dy in range(-obstacle_radius, obstacle_radius + 1):
                    ox, oy = gx + dx, gy + dy
                    if 0 <= ox < cols and 0 <= oy < rows:
                        if dx*dx + dy*dy <= obstacle_radius*obstacle_radius:
                            self.grid_dynamic[oy, ox] = 1
                            self.dynamic_obstacles.append((ox, oy))

            print(f"Obstacle added at {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.visualize_static()
            return

        # --------------------------------------------------------------------
        # NAVIGATION MODE
        # --------------------------------------------------------------------

        # Check if valid position
        if not self.is_valid(grid_pos):
            print(f"Invalid position clicked: {grid_pos} - position is occupied or out of bounds!")
            return

        if self.click_count == 0:
            # First click - set start
            self.start = grid_pos
            print(f"Start set to: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.click_count = 1
            self.visualize_static()

        elif self.click_count == 1:
            # Second click - set goal and run pathfinding
            self.goal = grid_pos
            print(f"Goal set to: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.click_count = 2

            # Run A* automatically
            self.run_astar()
            self.visualize_static()

            print("\nPress 'a' on the plot to see animated search (if path found)")
            print("Click again to reset and plan a new path")

        else:
            # Third click - reset for new planning
            print("\n" + "="*60)
            print("Resetting - click to set new start position")
            print("="*60)
            self.start = grid_pos
            self.goal = None
            self.path = []
            self.explored_nodes = []
            self.open_set_viz = []
            self.click_count = 1
            print(f"Start set to: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.visualize_static()

    def on_key(self, event):
        """
        Handle keyboard events for mode switching and actions.

        Keyboard commands:
            'a': Animate search process
            'o': Toggle obstacle placement mode
            'r': Run/replan pathfinding
            'c': Clear all and reset
            'm': Start/stop robot movement simulation

        Args:
            event: Matplotlib key_press_event
        """
        if event.key == 'a' and self.path:
            # Animate A* search
            print("\nStarting animation...")
            self.visualize_animated()

        elif event.key == 'a' and not self.path:
            print("\nNo path to animate! Set start and goal first.")

        elif event.key == 'o':
            # Toggle obstacle placement mode
            self.mode = 'add_obstacle' if self.mode == 'navigate' else 'navigate'
            mode_str = "OBSTACLE PLACEMENT" if self.mode == 'add_obstacle' else "NAVIGATION"
            print(f"\nMode: {mode_str}")
            if self.mode == 'add_obstacle':
                print("Click on the map to place obstacles")
            else:
                print("Click to set start/goal positions")

        elif event.key == 'r':
            # Run pathfinding
            self.run_astar()
            self.visualize_static()

        elif event.key == 'c':
            # Clear and reset
            print("\nClearing all...")
            self.start = None
            self.goal = None
            self.path = []
            self.explored_nodes = []
            self.click_count = 0
            self.dynamic_obstacles = []
            self.grid_dynamic = self.grid.copy()
            self.mode = 'navigate'
            self.robot_position = None
            self.robot_path_index = 0
            if self.robot_animation:
                self.robot_animation.event_source.stop()
                self.robot_animation = None
            self.animation_running = False
            self.visualize_static()
            print("Reset complete. Click to set start position.")

        elif event.key == 'm':
            # Toggle robot movement simulation
            if not self.path:
                print("\nNo path to simulate! Plan a path first.")
            elif self.animation_running:
                print("\nStopping robot simulation...")
                if self.robot_animation:
                    self.robot_animation.event_source.stop()
                    self.robot_animation = None
                self.animation_running = False
            else:
                print("\nStarting robot movement simulation...")
                self.start_robot_simulation()

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize_static(self):
        """
        Visualize current state without animation.

        Displays:
        - Static map with clearance zones
        - Dynamic obstacles (if any)
        - Explored nodes (if pathfinding ran)
        - Planned path (if found)
        - Start position (green circle)
        - Goal position (red star)
        - Robot position (purple circle, during simulation)
        """
        self.ax.clear()

        # --------------------------------------------------------------------
        # MAP DISPLAY
        # --------------------------------------------------------------------
        rows, cols = self.grid.shape
        extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        # Create colored map showing clearance zones
        # White = free space
        # Light red = safety clearance zone
        # Dark gray = original obstacles
        map_display = np.ones((rows, cols, 3))
        map_display[self.grid == 1] = VisualizerConstants.COLOR_CLEARANCE  # Safety zone
        map_display[self.grid_original == 1] = VisualizerConstants.COLOR_OBSTACLE  # Walls

        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.9, zorder=0)

        # --------------------------------------------------------------------
        # WORLD ORIGIN MARKER
        # --------------------------------------------------------------------
        self.ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3, label='World Origin (0,0)', zorder=10)
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)

        # --------------------------------------------------------------------
        # EXPLORED NODES
        # --------------------------------------------------------------------
        if self.explored_nodes:
            explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.explored_nodes]
            explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.explored_nodes]
            self.ax.scatter(explored_x, explored_y, c='cyan', s=5, alpha=0.3, label='Explored', zorder=1)

        # --------------------------------------------------------------------
        # PLANNED PATH
        # --------------------------------------------------------------------
        if self.path:
            path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
            path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
            self.ax.plot(path_x, path_y, 'b-',
                        linewidth=VisualizerConstants.PATH_LINE_WIDTH,
                        label=f'Path ({len(self.path)} nodes)',
                        alpha=0.8, zorder=3)
            # Add waypoint markers
            self.ax.scatter(path_x, path_y, c='blue', s=20, alpha=0.6, zorder=4)

        # --------------------------------------------------------------------
        # START AND GOAL
        # --------------------------------------------------------------------
        if self.start:
            start_world = self.grid_to_world(self.start[0], self.start[1])
            self.ax.plot(start_world[0], start_world[1], 'go',
                        markersize=VisualizerConstants.ROBOT_MARKER_SIZE,
                        label='Start',
                        markeredgecolor='black',
                        markeredgewidth=2,
                        zorder=5)

        if self.goal:
            goal_world = self.grid_to_world(self.goal[0], self.goal[1])
            self.ax.plot(goal_world[0], goal_world[1], 'r*',
                        markersize=VisualizerConstants.GOAL_MARKER_SIZE,
                        label='Goal',
                        markeredgecolor='black',
                        markeredgewidth=2,
                        zorder=5)

        # --------------------------------------------------------------------
        # ROBOT POSITION (DURING SIMULATION)
        # --------------------------------------------------------------------
        if self.robot_position:
            robot_world = self.grid_to_world(self.robot_position[0], self.robot_position[1])
            # Draw robot as a circle
            robot_circle = plt.Circle(robot_world, self.robot_radius,
                                     color='purple', alpha=0.7,
                                     zorder=6, label='Robot')
            self.ax.add_patch(robot_circle)
            # Draw robot center
            self.ax.plot(robot_world[0], robot_world[1], 'w+',
                        markersize=10, markeredgewidth=2, zorder=7)

        # --------------------------------------------------------------------
        # DYNAMIC OBSTACLES
        # --------------------------------------------------------------------
        if self.dynamic_obstacles:
            # Convert obstacle pixels to world coordinates
            obstacle_world_coords = []
            for ox, oy in self.dynamic_obstacles:
                if 0 <= ox < cols and 0 <= oy < rows:
                    wx, wy = self.grid_to_world(ox, oy)
                    obstacle_world_coords.append((wx, wy))

            # Draw each unique obstacle location
            unique_obstacles = set(obstacle_world_coords)
            for wx, wy in unique_obstacles:
                obstacle_circle = plt.Circle(
                    (wx, wy),
                    self.resolution/2,
                    color=VisualizerConstants.COLOR_DYNAMIC_OBS,
                    alpha=0.9,
                    edgecolor='darkorange',
                    linewidth=1,
                    zorder=8  # High zorder to be on top
                )
                self.ax.add_patch(obstacle_circle)

            # Add one label for all obstacles
            if unique_obstacles:
                first_obstacle = list(unique_obstacles)[0]
                self.ax.plot(first_obstacle[0], first_obstacle[1], 'o',
                           color=VisualizerConstants.COLOR_DYNAMIC_OBS,
                           markersize=8,
                           label=f'Dynamic Obstacles ({len(unique_obstacles)})',
                           zorder=8)

        # --------------------------------------------------------------------
        # AXIS CONFIGURATION
        # --------------------------------------------------------------------
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

        # Update title based on state
        title_suffix = ""
        if self.mode == 'add_obstacle':
            title_suffix = " [OBSTACLE MODE - Click to place obstacles]"

        if self.path:
            # Calculate path length
            path_length_m = sum([
                math.sqrt(
                    (self.grid_to_world(self.path[i][0], self.path[i][1])[0] -
                     self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[0])**2 +
                    (self.grid_to_world(self.path[i][0], self.path[i][1])[1] -
                     self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[1])**2
                )
                for i in range(len(self.path)-1)
            ])
            robot_status = " [ROBOT MOVING]" if self.animation_running else ""
            self.ax.set_title(
                f'Path Found! Length: {path_length_m:.2f}m, '
                f'Explored: {len(self.explored_nodes)} nodes{robot_status}{title_suffix}'
            )
        elif self.goal:
            self.ax.set_title(f'No path found!{title_suffix}')
        elif self.start:
            self.ax.set_title(f'Click to set Goal position{title_suffix}')
        else:
            self.ax.set_title(f'Click to set Start position{title_suffix}')

        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        plt.draw()

    def visualize_animated(self):
        """
        Animate the A* search process.

        Shows the progression of the A* algorithm:
        - Explored nodes (cyan) appearing as search progresses
        - Open set (yellow) showing frontier
        - Final path (blue) appearing when search completes
        """
        self.ax.clear()

        # --------------------------------------------------------------------
        # MAP DISPLAY
        # --------------------------------------------------------------------
        rows, cols = self.grid.shape
        extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        # Create colored map (simple black/white for animation)
        map_display = np.ones((rows, cols, 3))
        map_display[self.grid == 1] = [0, 0, 0]

        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.8)

        # --------------------------------------------------------------------
        # START AND GOAL
        # --------------------------------------------------------------------
        start_world = self.grid_to_world(self.start[0], self.start[1])
        goal_world = self.grid_to_world(self.goal[0], self.goal[1])
        self.ax.plot(start_world[0], start_world[1], 'go',
                    markersize=15, label='Start',
                    markeredgecolor='black', markeredgewidth=2)
        self.ax.plot(goal_world[0], goal_world[1], 'r*',
                    markersize=20, label='Goal',
                    markeredgecolor='black', markeredgewidth=2)

        # --------------------------------------------------------------------
        # ANIMATION SETUP
        # --------------------------------------------------------------------
        explored_scatter = self.ax.scatter([], [], c='cyan', s=10, alpha=0.5, label='Explored')
        open_scatter = self.ax.scatter([], [], c='yellow', s=10, alpha=0.5, label='Open Set')
        path_line, = self.ax.plot([], [], 'b-', linewidth=3, label='Path', alpha=0.7)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('A* Path Planner - Searching...')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # Animation function
        def animate(frame):
            """Update function for each animation frame"""
            if frame < len(self.explored_nodes):
                # Show explored nodes up to this frame
                explored_up_to_frame = self.explored_nodes[:frame+1]
                explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in explored_up_to_frame]
                explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in explored_up_to_frame]
                explored_scatter.set_offsets(np.c_[explored_x, explored_y])

                # Show current open set
                if frame < len(self.open_set_viz):
                    open_nodes = self.open_set_viz[frame]
                    if open_nodes:
                        open_x = [self.grid_to_world(gx, gy)[0] for gx, gy in open_nodes]
                        open_y = [self.grid_to_world(gx, gy)[1] for gx, gy in open_nodes]
                        open_scatter.set_offsets(np.c_[open_x, open_y])

                self.ax.set_title(f'A* Path Planner - Explored {frame+1}/{len(self.explored_nodes)} nodes')
            else:
                # Animation finished, show final path
                path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
                path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
                path_line.set_data(path_x, path_y)
                self.ax.set_title(f'A* Path Planner - Path Found! ({len(self.path)} waypoints)')

            return explored_scatter, open_scatter, path_line

        # Create animation
        frames = len(self.explored_nodes) + 20  # Extra frames to show final path
        interval = max(10, 2000 // len(self.explored_nodes))  # Adjust speed based on nodes

        anim = FuncAnimation(self.fig, animate, frames=frames, interval=interval,
                           blit=False, repeat=False)

        plt.draw()

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    def run(self, yaml_file, pgm_file):
        """
        Main function to run the interactive visualizer.

        Loads the map, sets up the matplotlib figure, connects event handlers,
        and starts the interactive session.

        Args:
            yaml_file: Path to map metadata YAML file
            pgm_file: Path to map image PGM file
        """
        # Load map
        self.load_map(yaml_file, pgm_file)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=VisualizerConstants.FIGURE_SIZE)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial visualization
        self.visualize_static()

        # Print instructions
        print("\n" + "="*60)
        print("D* Lite Interactive Visualizer with Dynamic Obstacles")
        print("="*60)
        print(f"Robot Configuration:")
        print(f"  - Robot radius: {self.robot_radius}m")
        print(f"  - Safety clearance: {self.safety_clearance}m")
        print(f"  - Total inflation: {self.robot_radius + self.safety_clearance}m")
        print("\nInstructions:")
        print("  1. Click on the map to set the START position (green circle)")
        print("  2. Click again to set the GOAL position (red star)")
        print("  3. The pathfinding algorithm will run and display the path")
        print("  4. Press 'o' to toggle OBSTACLE PLACEMENT mode")
        print("  5. Press 'm' to start ROBOT MOVEMENT simulation")
        print("     - Robot automatically detects obstacles 1m ahead")
        print("     - Automatically replans when obstacles detected")
        print("  6. Press 'r' to manually RUN/REPLAN pathfinding")
        print("  7. Press 'a' to see animated search process")
        print("  8. Press 'c' to CLEAR and start over")
        print("\nKeyboard Commands:")
        print("  'o' - Toggle obstacle placement mode")
        print("  'r' - Run/replan pathfinding")
        print("  'm' - Start/stop robot movement simulation")
        print("  'a' - Animate search process")
        print("  'c' - Clear all and reset")
        print("\nMap Legend:")
        print("  - White: Free space")
        print("  - Light red: Safety clearance zone")
        print("  - Dark gray: Static walls/obstacles")
        print("  - Orange: Dynamic obstacles (placed by you)")
        print("  - Green circle: Start position")
        print("  - Red star: Goal position")
        print("  - Purple circle: Robot (during simulation)")
        print("  - Cyan dots: Explored nodes")
        print("  - Blue line: Final path")
        print("="*60 + "\n")

        plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(args=None):
    """
    Main entry point for the visualizer.

    Command line usage:
        python3 visualizer.py
        python3 visualizer.py <yaml_file> <pgm_file>
        python3 visualizer.py <yaml_file> <pgm_file> <robot_radius>
        python3 visualizer.py <yaml_file> <pgm_file> <robot_radius> <safety_clearance>
    """
    import sys
    import os

    # Default parameters from constants
    robot_radius = VisualizerConstants.ROBOT_RADIUS
    safety_clearance = VisualizerConstants.SAFETY_CLEARANCE

    # Default map path
    yaml_file = os.path.expanduser('~/obstacle-avoidance-comps/ros2_ws/maze.yaml')
    pgm_file = os.path.expanduser('~/obstacle-avoidance-comps/ros2_ws/maze.pgm')

    # Parse command line arguments
    if len(sys.argv) >= 3:
        yaml_file = sys.argv[1]
        pgm_file = sys.argv[2]
        print(f"Using custom map: {yaml_file}, {pgm_file}")

    if len(sys.argv) >= 4:
        robot_radius = float(sys.argv[3])
        print(f"Using custom robot radius: {robot_radius}m")

    if len(sys.argv) >= 5:
        safety_clearance = float(sys.argv[4])
        print(f"Using custom safety clearance: {safety_clearance}m")

    # Create and run visualizer
    visualizer = DStarVisualizer(robot_radius=robot_radius, safety_clearance=safety_clearance)
    visualizer.run(yaml_file, pgm_file)


if __name__ == '__main__':
    main()
