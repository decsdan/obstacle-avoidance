#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import yaml
from PIL import Image
import heapq
import math

class AStarNavigator(Node):
    def __init__(self, robot_radius=0.22, safety_clearance=0.15):
        super().__init__('astar_navigator')

        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Map data
        self.grid = None
        self.grid_original = None  # Store original grid before inflation
        self.resolution = None
        self.origin = None

        # Robot parameters
        self.robot_radius = robot_radius  # TurtleBot4 radius (~0.22m)
        self.safety_clearance = safety_clearance  # Additional safety margin

        # Current state
        self.current_pose = None
        self.path = []
        self.current_waypoint_idx = 0

        # Control parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s
        self.position_tolerance = 0.1  # meters
        self.angle_tolerance = 0.1  # radians

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Load map
        self.load_map('/opt/ros/humble/share/turtlebot4_navigation/maps/maze.yaml', '/opt/ros/humble/share/turtlebot4_navigation/maps/maze.pgm')

        self.get_logger().info('A* Navigator initialized')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m, Safety clearance: {self.safety_clearance}m')
        self.get_logger().info(f'Total obstacle inflation: {self.robot_radius + self.safety_clearance}m')
        self.get_logger().info('Usage: navigator.navigate_to_goal(start_x, start_y, goal_x, goal_y)')
    
    def load_map(self, yaml_file, pgm_file):
        """Load map from files"""
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

            self.get_logger().info(f'Map loaded: {self.grid.shape}, resolution: {self.resolution}')
            self.get_logger().info(f'Obstacles inflated by {total_inflation}m ({int(total_inflation/self.resolution)} pixels)')
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')
    
    def inflate_obstacles(self, inflation_radius):
        """Inflate obstacles by specified radius (robot_radius + clearance)"""
        from scipy.ndimage import binary_dilation
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)
    
    def odom_callback(self, msg):
        """Update current robot pose from odometry"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }
    
    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """
        Main function to navigate from start to goal
        
        Args:
            start_x, start_y: Current position (meters)
            goal_x, goal_y: Goal position (meters)
        
        Returns:
            bool: True if path found and navigation started, False otherwise
        """
        self.get_logger().info(f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')
        
        # Plan path using A*
        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)
        
        if self.path:
            self.current_waypoint_idx = 0
            self.get_logger().info(f'✓ Path found with {len(self.path)} waypoints')
            self.print_path()
            return True
        else:
            self.get_logger().error('✗ No path found! Check if start/goal are valid and reachable')
            return False
    
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """Plan path using A*"""
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)
        
        self.get_logger().info(f'Grid coordinates: Start {start_grid}, Goal {goal_grid}')
        
        # Check if start/goal are valid
        if not self.is_valid(start_grid):
            self.get_logger().error(f'Start position ({start_x:.2f}, {start_y:.2f}) is invalid or occupied!')
            return []
        
        if not self.is_valid(goal_grid):
            self.get_logger().error(f'Goal position ({goal_x:.2f}, {goal_y:.2f}) is invalid or occupied!')
            return []
        
        # Run A* algorithm
        self.get_logger().info('Running A* pathfinding...')
        path_grid = self.astar(start_grid, goal_grid)
        
        if not path_grid:
            return []
        
        # Convert to world coordinates and simplify path
        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        simplified_path = self.simplify_path(path_world)
        
        return simplified_path
    
    def astar(self, start, goal):
        """A* pathfinding algorithm"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        rows, cols = self.grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        nodes_explored = 0
        
        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                self.get_logger().info(f'A* explored {nodes_explored} nodes')
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
        
        self.get_logger().error(f'A* failed after exploring {nodes_explored} nodes')
        return []  # No path found
    
    def is_valid(self, grid_pos):
        """Check if grid position is valid and free"""
        gx, gy = grid_pos
        rows, cols = self.grid.shape
        
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False
        
        return self.grid[gy, gx] == 0  # Note: grid is [row, col] = [y, x]
    
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
    
    def simplify_path(self, path, max_points=20):
        """Simplify path by keeping only key waypoints"""
        if len(path) <= max_points:
            return path
        
        step = len(path) // max_points
        simplified = [path[i] for i in range(0, len(path), step)]
        simplified.append(path[-1])  # Always include goal
        return simplified
    
    def print_path(self):
        """Print the planned path"""
        self.get_logger().info('Planned waypoints:')
        for i, (x, y) in enumerate(self.path):
            self.get_logger().info(f'  {i+1}. ({x:.2f}, {y:.2f})')
    
    def control_loop(self):
        """Main control loop - follows waypoints"""
        if not self.path or self.current_pose is None:
            return
        
        if self.current_waypoint_idx >= len(self.path):
            # Goal reached
            self.stop_robot()
            self.get_logger().info('🎯 Goal reached!')
            self.path = []
            return
        
        # Get current waypoint
        target = self.path[self.current_waypoint_idx]
        tx, ty = target
        
        # Calculate distance and angle to target
        dx = tx - self.current_pose['x']
        dy = ty - self.current_pose['y']
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - self.current_pose['theta'])
        
        # Create velocity command
        cmd = Twist()
        
        # If close enough to waypoint, move to next
        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached')
            return
        
        # First rotate towards target
        if abs(angle_diff) > self.angle_tolerance:
            cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
        else:
            # Then move forward
            cmd.linear.x = min(self.linear_speed, distance)
            # Small correction for angle while moving
            cmd.angular.z = 0.3 * angle_diff
        
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def get_current_position(self):
        """Get the current robot position from odometry"""
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return None
        return (self.current_pose['x'], self.current_pose['y'])
    
    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    # Default parameters - can be customized via environment variables
    import os
    robot_radius = float(os.getenv('ROBOT_RADIUS', '0.22'))  # TurtleBot4 radius
    safety_clearance = float(os.getenv('SAFETY_CLEARANCE', '0.15'))  # Safety margin

    rclpy.init(args=args)
    navigator = AStarNavigator(robot_radius=robot_radius, safety_clearance=safety_clearance)

    print("\n" + "="*60)
    print("TurtleBot4 A* Navigator")
    print("="*60)
    print(f"\nRobot Configuration:")
    print(f"  Robot radius: {robot_radius}m (set via ROBOT_RADIUS env var)")
    print(f"  Safety clearance: {safety_clearance}m (set via SAFETY_CLEARANCE env var)")
    print(f"  Total inflation: {robot_radius + safety_clearance}m")
    print("\nUsage Examples:")
    print("  1. Get current position:")
    print("     pos = navigator.get_current_position()")
    print("\n  2. Navigate to goal:")
    print("     navigator.navigate_to_goal(0.0, 0.0, 3.0, 2.0)")
    print("\n  3. Use current position:")
    print("     pos = navigator.get_current_position()")
    print("     if pos:")
    print("         navigator.navigate_to_goal(pos[0], pos[1], 3.0, 2.0)")
    print("\nTo customize clearance:")
    print("  ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run a_star run_a_star")
    print("="*60 + "\n")
    
    # Example: Wait for odometry, then navigate
    print("Waiting for odometry data...")
    while navigator.current_pose is None and rclpy.ok():
        rclpy.spin_once(navigator, timeout_sec=0.1)
    
    if navigator.current_pose:
        print(f"Current position: ({navigator.current_pose['x']:.2f}, {navigator.current_pose['y']:.2f})")
        
        # Example usage - you can modify these values
        goal_x = float(input("Enter goal X coordinate (meters): "))
        goal_y = float(input("Enter goal Y coordinate (meters): "))
        
        # Use current position as start
        start_x = navigator.current_pose['x']
        start_y = navigator.current_pose['y']
        
        # Plan and execute path
        if navigator.navigate_to_goal(start_x, start_y, goal_x, goal_y):
            print("\nNavigation started! Press Ctrl+C to stop.\n")
            try:
                rclpy.spin(navigator)
            except KeyboardInterrupt:
                print("\nStopping navigation...")
        else:
            print("\nNavigation failed!")
    
    navigator.stop_robot()
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()