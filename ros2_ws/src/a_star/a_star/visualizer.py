#!/usr/bin/env python3
# Originally authored by the 2025 Carleton Senior Capstone Project
# (see AUTHORS.md). Substantially rewritten by Daniel Scheider, 2026.
"""Interactive matplotlib visualizer for testing A* pathfinding with a static map."""

import heapq
import math
import os
import sys

import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
from PIL import Image
import matplotlib.pyplot as plt


class VisualizerConstants:
    """Configuration constants for the A* visualizer."""

    ROBOT_RADIUS = 0.22
    SAFETY_CLEARANCE = 0.15

    FIGURE_SIZE = (12, 10)
    PATH_LINE_WIDTH = 3
    ROBOT_MARKER_SIZE = 15
    GOAL_MARKER_SIZE = 20

    MIN_ANIMATION_INTERVAL = 10
    BASE_ANIMATION_TIME = 2000

    COLOR_FREE_SPACE = [1.0, 1.0, 1.0]
    COLOR_CLEARANCE = [1.0, 0.7, 0.7]
    COLOR_OBSTACLE = [0.2, 0.2, 0.2]


class AStarVisualizer:
    """Interactive A* pathfinding visualizer with animated search and static views."""

    def __init__(self, robot_radius=None, safety_clearance=None):
        """Initialize the visualizer with optional robot parameters."""
        if robot_radius is not None:
            self.robot_radius = robot_radius
        else:
            self.robot_radius = VisualizerConstants.ROBOT_RADIUS

        if safety_clearance is not None:
            self.safety_clearance = safety_clearance
        else:
            self.safety_clearance = VisualizerConstants.SAFETY_CLEARANCE

        self.grid = None
        self.grid_original = None
        self.resolution = None
        self.origin = None

        self.start = None
        self.goal = None
        self.path = []
        self.explored_nodes = []
        self.open_set_viz = []

        self.click_count = 0

        self.fig = None
        self.ax = None

    def load_map(self, yaml_file, pgm_file):
        """Load static map from YAML and PGM, then inflate obstacles."""
        try:
            with open(yaml_file, 'r') as f:
                map_data = yaml.safe_load(f)

            self.resolution = map_data['resolution']
            self.origin = map_data['origin']

            img = Image.open(pgm_file)
            occupancy_grid = np.array(img)

            self.grid = np.zeros_like(occupancy_grid)
            self.grid[occupancy_grid < 250] = 1
            self.grid[occupancy_grid >= 250] = 0
            self.grid = np.flipud(self.grid)
            self.grid_original = self.grid.copy()

            total_inflation = self.robot_radius + self.safety_clearance
            self.inflate_obstacles(total_inflation)

            print(f'Map loaded: {self.grid.shape} res={self.resolution}m '
                  f'inflation={total_inflation:.2f}m')
        except Exception as e:
            print(f'Failed to load map: {e}')

    def inflate_obstacles(self, inflation_radius):
        """Inflate obstacles by inflation_radius using morphological dilation."""
        from scipy.ndimage import binary_dilation

        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    def world_to_grid(self, x, y):
        """Convert world coordinates in meters to grid cell indices."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid cell indices to world coordinates in meters."""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    def is_valid(self, grid_pos):
        """Return True if grid_pos is in-bounds and free in the inflated grid."""
        gx, gy = grid_pos
        rows, cols = self.grid.shape
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False
        return self.grid[gy, gx] == 0

    def astar_with_visualization(self, start, goal):
        """A* search that records explored nodes and open set for animation."""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

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

            self.explored_nodes.append(current)
            self.open_set_viz.append([node[1] for node in open_set])

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print(f'A* found path, explored {nodes_explored} nodes')
                return path[::-1]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid(neighbor):
                    continue

                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f'A* failed after exploring {nodes_explored} nodes')
        return []

    def run_astar(self):
        """Run A* from current start to goal and print path statistics."""
        if self.start is None or self.goal is None:
            print('Please set both start and goal positions by clicking on the map')
            return

        print(f'Running A* from {self.start} to {self.goal}...')
        self.path = self.astar_with_visualization(self.start, self.goal)

        if self.path:
            path_world = [self.grid_to_world(gx, gy) for gx, gy in self.path]
            path_length = sum(
                math.sqrt(
                    (path_world[i][0] - path_world[i+1][0])**2 +
                    (path_world[i][1] - path_world[i+1][1])**2
                )
                for i in range(len(path_world) - 1)
            )
            print(f'Path found: {len(self.path)} waypoints, {path_length:.2f}m')
        else:
            print('No path found!')

    def on_click(self, event):
        """Handle mouse clicks: 1st sets start, 2nd sets goal + plans, 3rd resets."""
        if event.inaxes != self.ax:
            return

        x_click = event.xdata
        y_click = event.ydata
        grid_pos = self.world_to_grid(x_click, y_click)

        if not self.is_valid(grid_pos):
            print(f'Invalid position: {grid_pos} is occupied or out of bounds')
            return

        if self.click_count == 0:
            self.start = grid_pos
            print(f'Start: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})')
            self.click_count = 1
            self.visualize_static()

        elif self.click_count == 1:
            self.goal = grid_pos
            print(f'Goal: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})')
            self.click_count = 2
            self.run_astar()
            self.visualize_static()
            print("Press 'a' to animate search. Click again to reset.")

        else:
            self.start = grid_pos
            self.goal = None
            self.path = []
            self.explored_nodes = []
            self.open_set_viz = []
            self.click_count = 1
            print(f'Reset. Start: {grid_pos} (world: {x_click:.2f}, {y_click:.2f})')
            self.visualize_static()

    def on_key(self, event):
        """Handle key press: 'a' animates the search."""
        if event.key == 'a' and self.path:
            self.visualize_animated()
        elif event.key == 'a':
            print('No path to animate. Set start and goal first.')

    def visualize_static(self):
        """Render the current state: map, explored nodes, path, start, and goal."""
        self.ax.clear()

        rows, cols = self.grid.shape
        extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        map_display = np.ones((rows, cols, 3))
        map_display[self.grid == 1] = VisualizerConstants.COLOR_CLEARANCE
        map_display[self.grid_original == 1] = VisualizerConstants.COLOR_OBSTACLE
        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.9)

        self.ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3,
                     label='World Origin (0,0)', zorder=10)
        self.ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)

        if self.explored_nodes:
            explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.explored_nodes]
            explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.explored_nodes]
            self.ax.scatter(explored_x, explored_y, c='cyan', s=5, alpha=0.3,
                            label='Explored', zorder=1)

        if self.path:
            path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
            path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
            self.ax.plot(path_x, path_y, 'b-',
                         linewidth=VisualizerConstants.PATH_LINE_WIDTH,
                         label=f'Path ({len(self.path)} nodes)',
                         alpha=0.8, zorder=3)
            self.ax.scatter(path_x, path_y, c='blue', s=20, alpha=0.6, zorder=4)

        if self.start:
            start_world = self.grid_to_world(self.start[0], self.start[1])
            self.ax.plot(start_world[0], start_world[1], 'go',
                         markersize=VisualizerConstants.ROBOT_MARKER_SIZE,
                         label='Start', markeredgecolor='black', markeredgewidth=2, zorder=5)

        if self.goal:
            goal_world = self.grid_to_world(self.goal[0], self.goal[1])
            self.ax.plot(goal_world[0], goal_world[1], 'r*',
                         markersize=VisualizerConstants.GOAL_MARKER_SIZE,
                         label='Goal', markeredgecolor='black', markeredgewidth=2, zorder=5)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

        if self.path:
            path_length_m = sum(
                math.sqrt(
                    (self.grid_to_world(self.path[i][0], self.path[i][1])[0] -
                     self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[0])**2 +
                    (self.grid_to_world(self.path[i][0], self.path[i][1])[1] -
                     self.grid_to_world(self.path[i+1][0], self.path[i+1][1])[1])**2
                )
                for i in range(len(self.path)-1)
            )
            self.ax.set_title(
                f'A* Path Found! Length: {path_length_m:.2f}m, '
                f'Explored: {len(self.explored_nodes)} nodes'
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
        """Animate the A* search showing explored nodes, open set, and final path."""
        self.ax.clear()

        rows, cols = self.grid.shape
        extent = [
            self.origin[0],
            self.origin[0] + cols * self.resolution,
            self.origin[1],
            self.origin[1] + rows * self.resolution
        ]

        map_display = np.ones((rows, cols, 3))
        map_display[self.grid == 1] = [0, 0, 0]
        self.ax.imshow(map_display, extent=extent, origin='lower', alpha=0.8)

        start_world = self.grid_to_world(self.start[0], self.start[1])
        goal_world = self.grid_to_world(self.goal[0], self.goal[1])
        self.ax.plot(start_world[0], start_world[1], 'go',
                     markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        self.ax.plot(goal_world[0], goal_world[1], 'r*',
                     markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)

        explored_scatter = self.ax.scatter([], [], c='cyan', s=10, alpha=0.5, label='Explored')
        open_scatter = self.ax.scatter([], [], c='yellow', s=10, alpha=0.5, label='Open Set')
        path_line, = self.ax.plot([], [], 'b-', linewidth=3, label='Path', alpha=0.7)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('A* Path Planner - Searching...')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        def animate(frame):
            if frame < len(self.explored_nodes):
                explored_up_to_frame = self.explored_nodes[:frame+1]
                explored_x = [self.grid_to_world(gx, gy)[0] for gx, gy in explored_up_to_frame]
                explored_y = [self.grid_to_world(gx, gy)[1] for gx, gy in explored_up_to_frame]
                explored_scatter.set_offsets(np.c_[explored_x, explored_y])

                if frame < len(self.open_set_viz):
                    open_nodes = self.open_set_viz[frame]
                    if open_nodes:
                        open_x = [self.grid_to_world(gx, gy)[0] for gx, gy in open_nodes]
                        open_y = [self.grid_to_world(gx, gy)[1] for gx, gy in open_nodes]
                        open_scatter.set_offsets(np.c_[open_x, open_y])

                self.ax.set_title(
                    f'A* - Explored {frame+1}/{len(self.explored_nodes)} nodes')
            else:
                path_x = [self.grid_to_world(gx, gy)[0] for gx, gy in self.path]
                path_y = [self.grid_to_world(gx, gy)[1] for gx, gy in self.path]
                path_line.set_data(path_x, path_y)
                self.ax.set_title(f'A* Path Found! ({len(self.path)} waypoints)')

            return explored_scatter, open_scatter, path_line

        frames = len(self.explored_nodes) + 20
        interval = max(
            VisualizerConstants.MIN_ANIMATION_INTERVAL,
            VisualizerConstants.BASE_ANIMATION_TIME // len(self.explored_nodes)
        )
        FuncAnimation(self.fig, animate, frames=frames, interval=interval,
                      blit=False, repeat=False)
        plt.draw()

    def run(self, yaml_file, pgm_file):
        """Load map, set up figure, connect events, and start interactive session."""
        self.load_map(yaml_file, pgm_file)

        self.fig, self.ax = plt.subplots(figsize=VisualizerConstants.FIGURE_SIZE)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.visualize_static()

        print(f'A* Visualizer | radius={self.robot_radius}m clearance={self.safety_clearance}m')
        print('  1. Click to set start (green), click again to set goal (red)')
        print("  2. Path runs automatically. Press 'a' to animate search.")
        print('  3. Click a third time to reset.')

        plt.show()


def main(args=None):
    """Entry point for the A* visualizer."""
    robot_radius = VisualizerConstants.ROBOT_RADIUS
    safety_clearance = VisualizerConstants.SAFETY_CLEARANCE

    yaml_file = '~/obstacle-avoidance-comps/ros2_ws/olin304-308.yaml'
    pgm_file = '~/obstacle-avoidance-comps/ros2_ws/olin304-308.pgm'

    if len(sys.argv) >= 3:
        yaml_file = sys.argv[1]
        pgm_file = sys.argv[2]

    if len(sys.argv) >= 4:
        robot_radius = float(sys.argv[3])

    if len(sys.argv) >= 5:
        safety_clearance = float(sys.argv[4])

    visualizer = AStarVisualizer(robot_radius=robot_radius, safety_clearance=safety_clearance)
    visualizer.run(yaml_file, pgm_file)


if __name__ == '__main__':
    main()
