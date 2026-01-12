#!/usr/bin/env python3

"""
Interactive JPS Path Visualizer for TurtleBot4

This visualizer provides a simple matplotlib-based interface for testing
the Jump Point Search (JPS) pathfinding algorithm with interactive start/goal selection.

Features:
    - Interactive start/goal selection via mouse clicks
    - Jump Point Search with jump point visualization
    - Animated search process showing algorithm progression
    - Real-time path length calculation
    - Clearance zone visualization

Usage:
    Run directly: python3 jps_visualizer.py
    With custom map: python3 jps_visualizer.py <yaml_file> <pgm_file>
    With custom robot params: python3 jps_visualizer.py <yaml> <pgm> <radius> <clearance>

Interactive Controls:
    - Click: Set start (1st click), goal (2nd click), reset (3rd+ click)
    - 'a': Animate search process

Architecture:
    - Pure JPS implementation (no dynamic obstacles)
    - Static map with obstacle inflation
    - 8-connected grid movement with jump points
    - Euclidean distance heuristic
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
    """Configuration constants for the JPS visualizer"""

    # Robot Physical Parameters (meters)
    ROBOT_RADIUS = 0.22          # TurtleBot4 radius
    SAFETY_CLEARANCE = 0.15      # Additional safety margin

    # Visualization
    FIGURE_SIZE = (12, 10)       # Figure dimensions (width, height)
    PATH_LINE_WIDTH = 3          # Width of path line
    ROBOT_MARKER_SIZE = 15       # Size of start marker
    GOAL_MARKER_SIZE = 20        # Size of goal marker
    JUMP_POINT_SIZE = 8          # Size of jump point markers

    # Animation
    MIN_ANIMATION_INTERVAL = 10  # Minimum milliseconds between frames
    BASE_ANIMATION_TIME = 2000   # Total animation time in milliseconds

    # Colors (RGB)
    COLOR_FREE_SPACE = [1.0, 1.0, 1.0]      # White
    COLOR_CLEARANCE = [1.0, 0.7, 0.7]        # Light red
    COLOR_OBSTACLE = [0.2, 0.2, 0.2]         # Dark gray
    COLOR_JUMP_POINTS = [1.0, 0.5, 0.0]      # Orange for jump points


# ============================================================================
# MAIN VISUALIZER CLASS
# ============================================================================

class JPSVisualizer:
    """
    Interactive JPS pathfinding visualizer.

    Provides a simple interface for testing JPS pathfinding with
    interactive start/goal selection and animated search visualization.
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
        self.jump_points = []           # Jump points discovered during search

        # ====================================================================
        # INITIALIZATION - Interactive State
        # ====================================================================

        self.click_count = 0            # Track clicks: 0=none, 1=start set, 2=goal set

        # ====================================================================
        # INITIALIZATION - Visualization
        # ====================================================================

        self.fig = None                 # Matplotlib figure
        self.ax = None                  # Matplotlib axis

    # ========================================================================
    # MAP LOADING AND PROCESSING
    # ========================================================================

    def load_map(self, yaml_file, pgm_file):
        """
        Load map from YAML and PGM files.

        Loads the static map, applies coordinate transforms to match Gazebo,
        and inflates obstacles by robot radius + safety clearance.

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

            # Print configuration
            print(f'Map loaded: {self.grid.shape}, resolution: {self.resolution}')
            print(f'Robot radius: {self.robot_radius}m, Safety clearance: {self.safety_clearance}m')
            print(f'Total obstacle inflation: {total_inflation}m ({int(total_inflation/self.resolution)} pixels)')
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

    def is_valid(self, grid_pos):
        """
        Check if a grid position is valid and free of obstacles.

        Args:
            grid_pos: Tuple (grid_x, grid_y) to check

        Returns:
            bool: True if position is valid and free, False otherwise
        """
        gx, gy = grid_pos
        rows, cols = self.grid.shape

        # Check bounds
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        # Check if free (0 = free, 1 = occupied)
        return self.grid[gy, gx] == 0  # Note: grid is [row, col] = [y, x]

    # ========================================================================
    # JPS PATHFINDING ALGORITHM
    # ========================================================================

    def jps_with_visualization(self, start, goal):
        """
        Jump Point Search algorithm with step-by-step visualization data.

        Finds the shortest path from start to goal using JPS search with
        Euclidean distance heuristic. Supports 8-connected grid movement
        with jump points.

        Args:
            start: Start position as (grid_x, grid_y)
            goal: Goal position as (grid_x, grid_y)

        Returns:
            list: Path as list of (grid_x, grid_y) tuples, or empty list if no path
        """
        # Heuristic: Euclidean distance
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        rows, cols = self.grid.shape

        # Initialize JPS data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        # Reset visualization data
        self.explored_nodes = []
        self.open_set_viz = []
        self.jump_points = []
        nodes_explored = 0

        # JPS main loop
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
                print(f'JPS explored {nodes_explored} nodes, found {len(self.jump_points)} jump points')
                return path[::-1]  # Reverse to get start->goal order

            # Get jump point successors instead of all neighbors
            for successor in self.get_successors_with_viz(current, start, goal):
                # Calculate distance between current and successor
                dx = successor[0] - current[0]
                dy = successor[1] - current[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                tentative_g = g_score[current] + distance

                # Update if better path found
                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f_score[successor] = tentative_g + heuristic(successor, goal)
                    heapq.heappush(open_set, (f_score[successor], successor))

        # No path found
        print(f'JPS failed after exploring {nodes_explored} nodes, found {len(self.jump_points)} jump points')
        return []

    def get_successors_with_viz(self, node, start, goal):
        """
        Get jump point successors for a given node with visualization.

        Args:
            node: Current node (grid_x, grid_y)
            start: Start position (for hybrid validation)
            goal: Goal position

        Returns:
            list: List of jump point successors
        """
        successors = []
        
        # Check all 8 directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            
            # Try to jump in this direction
            jump_point = self.jump_with_viz(node, (dx, dy), start, goal)
            
            if jump_point is not None:
                successors.append(jump_point)
                
        return successors

    def jump_with_viz(self, node, direction, start, goal, depth=0):
        """
        Jump recursively in a direction until finding a jump point with visualization.
        
        Args:
            node: Current node (grid_x, grid_y)
            direction: Direction (dx, dy) to jump
            start: Start position (for hybrid validation)
            goal: Goal position
            depth: Recursion depth counter (for preventing infinite recursion)

        Returns:
            tuple: Jump point coordinates or None if blocked
        """
        # Prevent infinite recursion
        if depth > 50:  # Reduced from 100
            # print(f"DEBUG: Max recursion depth reached at {node}")
            return None
            
        dx, dy = direction
        next_node = (node[0] + dx, node[1] + dy)
        
        # Check if next node is valid
        if not self.is_valid(next_node):
            return None
            
        # If we reached the goal, it's a jump point
        if next_node == goal:
            if next_node not in self.jump_points:
                self.jump_points.append(next_node)
            return next_node
            
        # Check for forced neighbors
        has_forced_neighbor = False
        
        # Diagonal movement
        if dx != 0 and dy != 0:
            # Check natural neighbors
            if (self.is_valid((next_node[0] + dx, next_node[1])) or 
                self.is_valid((next_node[0], next_node[1] + dy))):
                # Check for forced neighbor
                if (self.is_valid((next_node[0] - dx, next_node[1])) and 
                    not self.is_valid((next_node[0] - dx, next_node[1] + dy))):
                    has_forced_neighbor = True
                elif (self.is_valid((next_node[0], next_node[1] - dy)) and 
                      not self.is_valid((next_node[0] + dx, next_node[1] - dy))):
                    has_forced_neighbor = True
        else:
            # Horizontal movement
            if dx != 0:
                # Check for forced neighbor above
                if (self.is_valid((next_node[0], next_node[1] + 1)) and 
                    not self.is_valid((next_node[0] + dx, next_node[1] + 1))):
                    has_forced_neighbor = True
                # Check for forced neighbor below
                elif (self.is_valid((next_node[0], next_node[1] - 1)) and 
                      not self.is_valid((next_node[0] + dx, next_node[1] - 1))):
                    has_forced_neighbor = True
            # Vertical movement
            else:
                # Check for forced neighbor right
                if (self.is_valid((next_node[0] + 1, next_node[1])) and 
                    not self.is_valid((next_node[0] + 1, next_node[1] + dy))):
                    has_forced_neighbor = True
                # Check for forced neighbor left
                elif (self.is_valid((next_node[0] - 1, next_node[1])) and 
                      not self.is_valid((next_node[0] - 1, next_node[1] + dy))):
                    has_forced_neighbor = True
        
        # If forced neighbor found or reached goal, this is a jump point
        if has_forced_neighbor:
            if next_node not in self.jump_points:
                self.jump_points.append(next_node)
            return next_node
        
        # Continue jumping in the same direction
        return self.jump_with_viz(next_node, direction, start, goal, depth+1)

    def run_jps(self):
        """
        Run JPS algorithm and store results.

        Executes pathfinding from current start to goal. Updates internal
        path state and prints path statistics.
        """
        if self.start is None or self.goal is None:
            print("Please set both start and goal positions by clicking on the map")
            return

        print(f"\nRunning Jump Point Search from {self.start} to {self.goal}...")
        self.path = self.jps_with_visualization(self.start, self.goal)

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
            print(f"Jump points discovered: {len(self.jump_points)}")
        else:
            print("No path found!")

    # ========================================================================
    # MOUSE AND KEYBOARD EVENT HANDLERS
    # ========================================================================

    def on_click(self, event):
        """
        Handle mouse clicks for start/goal selection.

        Behavior:
        - First click: Set start position
        - Second click: Set goal and run pathfinding
        - Third+ click: Reset and set new start

        Args:
            event: Matplotlib button_press_event
        """
        if event.inaxes != self.ax:
            return

        # Convert click coordinates to grid coordinates
        x_click = event.xdata
        y_click = event.ydata
        grid_pos = self.world_to_grid(x_click, y_click)

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

            # Run JPS automatically
            self.run_jps()
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
            self.jump_points = []
            self.click_count = 1
            print(f"Start set to: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.visualize_static()

    def on_key(self, event):
        """
        Handle keyboard events.

        Keyboard commands:
            'a': Animate search process

        Args:
            event: Matplotlib key_press_event
        """
        if event.key == 'a' and self.path:
            print("\nStarting animation...")
            self.visualize_animated()
        elif event.key == 'a' and not self.path:
            print("\nNo path to animate! Set start and goal first.")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize_static(self):
        """
        Visualize current state without animation.

        Displays:
        - Static map with clearance zones
        - Explored nodes (if pathfinding ran)
        - Jump points (orange)
        - Planned path (if found)
        - Start position (green circle)
        - Goal position (red star)
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

        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.9)

        # --------------------------------------------------------------------
        # WORLD ORIGIN MARKER
        # --------------------------------------------------------------------
        self.ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3,
                    label='World Origin (0,0)', zorder=10)
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)

        # --------------------------------------------------------------------
        # JUMP POINTS
        # --------------------------------------------------------------------
        if self.jump_points:
            jump_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.jump_points]
            jump_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.jump_points]
            self.ax.scatter(jump_x, jump_y, c=VisualizerConstants.COLOR_JUMP_POINTS, 
                          s=VisualizerConstants.JUMP_POINT_SIZE, alpha=0.7,
                          label=f'Jump Points ({len(self.jump_points)})', zorder=2)

        # --------------------------------------------------------------------
        # EXPLORED NODES
        # --------------------------------------------------------------------
        if self.explored_nodes:
            explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.explored_nodes]
            explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.explored_nodes]
            self.ax.scatter(explored_x, explored_y, c='cyan', s=5, alpha=0.3,
                          label='Explored', zorder=1)

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
        # AXIS CONFIGURATION
        # --------------------------------------------------------------------
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

        # Update title based on state
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
            self.ax.set_title(
                f'JPS Path Found! Length: {path_length_m:.2f}m, '
                f'Explored: {len(self.explored_nodes)} nodes, '
                f'Jump Points: {len(self.jump_points)}'
            )
        elif self.goal:
            self.ax.set_title('No path found!')
        elif self.start:
            self.ax.set_title('Click to set Goal position')
        else:
            self.ax.set_title('Click to set Start position')

        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        plt.draw()

    def visualize_animated(self):
        """
        Animate the JPS search process.

        Shows the progression of the JPS algorithm:
        - Explored nodes (cyan) appearing as search progresses
        - Jump points (orange) appearing as discovered
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
        jump_scatter = self.ax.scatter([], [], c=VisualizerConstants.COLOR_JUMP_POINTS, 
                                      s=VisualizerConstants.JUMP_POINT_SIZE, alpha=0.7, label='Jump Points')
        open_scatter = self.ax.scatter([], [], c='yellow', s=10, alpha=0.5, label='Open Set')
        path_line, = self.ax.plot([], [], 'b-', linewidth=3, label='Path', alpha=0.7)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('JPS Path Planner - Searching...')
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

                # Show jump points up to this frame
                jump_points_up_to_frame = [p for p in self.jump_points if p in explored_up_to_frame]
                if jump_points_up_to_frame:
                    jump_x = [self.grid_to_world(gx, gy)[0] for gx, gy in jump_points_up_to_frame]
                    jump_y = [self.grid_to_world(gx, gy)[1] for gx, gy in jump_points_up_to_frame]
                    jump_scatter.set_offsets(np.c_[jump_x, jump_y])

                # Show current open set
                if frame < len(self.open_set_viz):
                    open_nodes = self.open_set_viz[frame]
                    if open_nodes:
                        open_x = [self.grid_to_world(gx, gy)[0] for gx, gy in open_nodes]
                        open_y = [self.grid_to_world(gx, gy)[1] for gx, gy in open_nodes]
                        open_scatter.set_offsets(np.c_[open_x, open_y])

                self.ax.set_title(f'JPS Path Planner - Explored {frame+1}/{len(self.explored_nodes)} nodes')
            else:
                # Animation finished, show final path
                path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
                path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
                path_line.set_data(path_x, path_y)
                self.ax.set_title(f'JPS Path Planner - Path Found! ({len(self.path)} waypoints)')

            return explored_scatter, jump_scatter, open_scatter, path_line

        # Create animation
        frames = len(self.explored_nodes) + 20  # Extra frames to show final path
        interval = max(
            VisualizerConstants.MIN_ANIMATION_INTERVAL,
            VisualizerConstants.BASE_ANIMATION_TIME // len(self.explored_nodes)
        )

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
        print("JPS Interactive Visualizer")
        print("="*60)
        print(f"Robot Configuration:")
        print(f"  - Robot radius: {self.robot_radius}m")
        print(f"  - Safety clearance: {self.safety_clearance}m")
        print(f"  - Total inflation: {self.robot_radius + self.safety_clearance}m")
        print("\nInstructions:")
        print("  1. Click on the map to set the START position (green circle)")
        print("  2. Click again to set the GOAL position (red star)")
        print("  3. The JPS algorithm will run and display the path immediately")
        print("  4. Press 'a' to see animated search process (optional)")
        print("  5. Click again to reset and plan a new path")
        print("\nMap Legend:")
        print("  - White: Free space")
        print("  - Light red: Safety clearance zone (robot + clearance)")
        print("  - Dark gray: Walls/obstacles")
        print("  - Green circle: Start position")
        print("  - Red star: Goal position")
        print("  - Cyan dots: Explored nodes")
        print("  - Orange dots: Jump points")
        print("  - Yellow dots: Open set (frontier) - in animation")
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
        python3 jps_visualizer.py
        python3 jps_visualizer.py <yaml_file> <pgm_file>
        python3 jps_visualizer.py <yaml_file> <pgm_file> <robot_radius>
        python3 jps_visualizer.py <yaml_file> <pgm_file> <robot_radius> <safety_clearance>
    """
    import sys
    import os

    # Default parameters from constants
    robot_radius = VisualizerConstants.ROBOT_RADIUS
    safety_clearance = VisualizerConstants.SAFETY_CLEARANCE

    # Default map path
    yaml_file = os.path.expanduser('~/code/obstacle-avoidance-comps/ros2_ws/maze_slamed.yaml')
    pgm_file = os.path.expanduser('~/code/obstacle-avoidance-comps/ros2_ws/maze_slamed.pgm')

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
    visualizer = JPSVisualizer(robot_radius=robot_radius, safety_clearance=safety_clearance)
    visualizer.run(yaml_file, pgm_file)


if __name__ == '__main__':
    main()