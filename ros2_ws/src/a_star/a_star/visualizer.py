#!/usr/bin/env python3

import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import math

class AStarVisualizer:
    def __init__(self, robot_radius=0.22, safety_clearance=0.15):
        # Map data
        self.grid = None
        self.grid_original = None  # Store original grid for visualization
        self.resolution = None
        self.origin = None

        # Robot parameters
        self.robot_radius = robot_radius  # TurtleBot4 radius (~0.22m)
        self.safety_clearance = safety_clearance  # Additional safety margin

        # Visualization data
        self.start = None
        self.goal = None
        self.path = []
        self.explored_nodes = []
        self.open_set_viz = []

        # Interactive state
        self.click_count = 0

        # Figure and axis
        self.fig = None
        self.ax = None

    def load_map(self, yaml_file, pgm_file):
        """Load map from files - same as run_a_star.py"""
        try:
            with open(yaml_file, 'r') as f:
                map_data = yaml.safe_load(f)

            self.resolution = map_data['resolution']
            self.origin = map_data['origin']

            img = Image.open(pgm_file)
            occupancy_grid = np.array(img)

            # Convert to binary (0=free, 1=occupied)
            self.grid = np.zeros_like(occupancy_grid)
            self.grid[occupancy_grid < 250] = 1  # Occupied/unknown
            self.grid[occupancy_grid >= 250] = 0  # Free

            # Flip map vertically to match Gazebo coordinate system (y-axis flip)
            self.grid = np.flipud(self.grid)

            # Store original grid before inflation
            self.grid_original = self.grid.copy()

            # Inflate obstacles for safety (robot radius + safety clearance)
            total_inflation = self.robot_radius + self.safety_clearance
            self.inflate_obstacles(total_inflation)

            print(f'Map loaded: {self.grid.shape}, resolution: {self.resolution}')
            print(f'Robot radius: {self.robot_radius}m, Safety clearance: {self.safety_clearance}m')
            print(f'Total obstacle inflation: {total_inflation}m ({int(total_inflation/self.resolution)} pixels)')
        except Exception as e:
            print(f'Failed to load map: {e}')

    def inflate_obstacles(self, inflation_radius):
        """Inflate obstacles by specified radius (robot_radius + clearance)"""
        from scipy.ndimage import binary_dilation
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    def is_valid(self, grid_pos):
        """Check if grid position is valid and free"""
        gx, gy = grid_pos
        rows, cols = self.grid.shape

        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        return self.grid[gy, gx] == 0  # Note: grid is [row, col] = [y, x]

    def astar_with_visualization(self, start, goal):
        """A* pathfinding algorithm with step-by-step visualization data"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        rows, cols = self.grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        self.explored_nodes = []
        self.open_set_viz = []

        nodes_explored = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

            # Store for visualization
            self.explored_nodes.append(current)
            self.open_set_viz.append([node[1] for node in open_set])

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print(f'A* explored {nodes_explored} nodes')
                return path[::-1]

            # 8-connected neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid(neighbor):
                    continue

                # Diagonal moves cost more
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f'A* failed after exploring {nodes_explored} nodes')
        return []  # No path found

    def run_astar(self):
        """Run A* algorithm and store results"""
        if self.start is None or self.goal is None:
            print("Please set both start and goal positions by clicking on the map")
            return

        print(f"\nRunning A* from {self.start} to {self.goal}...")
        self.path = self.astar_with_visualization(self.start, self.goal)

        if self.path:
            print(f"Path found with {len(self.path)} waypoints!")
            # Convert to world coordinates for display
            path_world = [self.grid_to_world(gx, gy) for gx, gy in self.path]
            print(f"Path length: {sum([math.sqrt((path_world[i][0]-path_world[i+1][0])**2 + (path_world[i][1]-path_world[i+1][1])**2) for i in range(len(path_world)-1)]):.2f} meters")
        else:
            print("No path found!")

    def on_click(self, event):
        """Handle mouse clicks to set start/goal"""
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
            # Second click - set goal
            self.goal = grid_pos
            print(f"Goal set to: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})")
            self.click_count = 2

            # Run A* automatically
            self.run_astar()

            # Visualize result - always use static for immediate feedback
            self.visualize_static()

            print("\nPress 'a' on the plot to see animated search (if path found)")
            print("Click again to reset and plan a new path")
        else:
            # Reset for new planning
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
        """Handle keyboard events"""
        if event.key == 'a' and self.path:
            print("\nStarting animation...")
            self.visualize_animated()
        elif event.key == 'a' and not self.path:
            print("\nNo path to animate! Set start and goal first.")

    def visualize_static(self):
        """Visualize current state without animation"""
        self.ax.clear()

        # Display map
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

        # Safety clearance zone (inflated area but not original obstacle)
        map_display[self.grid == 1] = [1.0, 0.7, 0.7]  # Light red for clearance

        # Original obstacles
        map_display[self.grid_original == 1] = [0.2, 0.2, 0.2]  # Dark gray for walls

        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.9)

        # Draw origin marker (0,0) with crosshairs
        self.ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3, label='World Origin (0,0)', zorder=10)
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)

        # Draw explored nodes if they exist
        if self.explored_nodes:
            explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.explored_nodes]
            explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.explored_nodes]
            self.ax.scatter(explored_x, explored_y, c='cyan', s=5, alpha=0.3, label='Explored', zorder=1)

        # Draw path if exists
        if self.path:
            path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
            path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, label=f'Path ({len(self.path)} nodes)', alpha=0.8, zorder=3)
            # Add waypoint markers
            self.ax.scatter(path_x, path_y, c='blue', s=20, alpha=0.6, zorder=4)

        # Draw start
        if self.start:
            start_world = self.grid_to_world(self.start[0], self.start[1])
            self.ax.plot(start_world[0], start_world[1], 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2, zorder=5)

        # Draw goal
        if self.goal:
            goal_world = self.grid_to_world(self.goal[0], self.goal[1])
            self.ax.plot(goal_world[0], goal_world[1], 'r*', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2, zorder=5)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

        # Update title based on state
        if self.path:
            path_length_m = sum([math.sqrt((self.grid_to_world(self.path[i][0], self.path[i][1])[0] - self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[0])**2 +
                                           (self.grid_to_world(self.path[i][0], self.path[i][1])[1] - self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[1])**2)
                                 for i in range(len(self.path)-1)])
            self.ax.set_title(f'A* Path Found! Length: {path_length_m:.2f}m, Explored: {len(self.explored_nodes)} nodes')
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
        """Animate the A* search process"""
        self.ax.clear()

        # Display map
        rows, cols = self.grid.shape
        extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        # Create colored map
        map_display = np.ones((rows, cols, 3))
        map_display[self.grid == 1] = [0, 0, 0]

        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.8)

        # Draw start and goal
        start_world = self.grid_to_world(self.start[0], self.start[1])
        goal_world = self.grid_to_world(self.goal[0], self.goal[1])
        self.ax.plot(start_world[0], start_world[1], 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        self.ax.plot(goal_world[0], goal_world[1], 'r*', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)

        # Prepare for animation
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

    def run(self, yaml_file, pgm_file):
        """Main function to run the visualizer"""
        # Load map
        self.load_map(yaml_file, pgm_file)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial visualization
        self.visualize_static()

        print("\n" + "="*60)
        print("A* Interactive Visualizer")
        print("="*60)
        print(f"Robot Configuration:")
        print(f"  - Robot radius: {self.robot_radius}m")
        print(f"  - Safety clearance: {self.safety_clearance}m")
        print(f"  - Total inflation: {self.robot_radius + self.safety_clearance}m")
        print("\nInstructions:")
        print("  1. Click on the map to set the START position (green circle)")
        print("  2. Click again to set the GOAL position (red star)")
        print("  3. The A* algorithm will run and display the path immediately")
        print("  4. Press 'a' to see animated search process (optional)")
        print("  5. Click again to reset and plan a new path")
        print("\nMap Legend:")
        print("  - White: Free space")
        print("  - Light red: Safety clearance zone (robot + clearance)")
        print("  - Dark gray: Walls/obstacles")
        print("  - Green circle: Start position")
        print("  - Red star: Goal position")
        print("  - Cyan dots: Explored nodes")
        print("  - Yellow dots: Open set (frontier) - in animation")
        print("  - Blue line: Final path")
        print("="*60 + "\n")

        plt.show()


def main(args=None):
    """Main entry point"""
    import sys

    # Default parameters
    robot_radius = 0.22  # TurtleBot4 radius in meters
    safety_clearance = 0.15  # Additional safety margin in meters

    # Default map path (same as run_a_star.py)
    yaml_file = '/opt/ros/jazzy/share/turtlebot4_navigation/maps/maze.yaml'
    pgm_file = '/opt/ros/jazzy/share/turtlebot4_navigation/maps/maze.pgm'

    # Parse command line arguments
    # Usage: visualizer [yaml_file] [pgm_file] [robot_radius] [safety_clearance]
    if len(sys.argv) >= 3:
        pgm_file = sys.argv[2]
        yaml_file = sys.argv[1]
        print(f"Using custom map: {yaml_file}, {pgm_file}")

    if len(sys.argv) >= 4:
        robot_radius = float(sys.argv[3])
        print(f"Using custom robot radius: {robot_radius}m")

    if len(sys.argv) >= 5:
        safety_clearance = float(sys.argv[4])
        print(f"Using custom safety clearance: {safety_clearance}m")

    visualizer = AStarVisualizer(robot_radius=robot_radius, safety_clearance=safety_clearance)
    visualizer.run(yaml_file, pgm_file)


if __name__ == '__main__':
    main()
