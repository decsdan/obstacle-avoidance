import rclpy
import rclpy.time
import rclpy.duration
import numpy as np
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from dwa_package import distanceGrid as dg
#import nav2_amcl #TODO need to sudo import nav2 for this, to get properly localized x y coords (not just odometry)                                   

class DWA(Node):
    """
    Dynamic Window Approach (DWA) local planner for TurtleBot4.
    
    Each control cycle, we compute the range of velocities reachable given
    current speed + acceleration limits (the "dynamic window"), simulate a
    short trajectory for each (v, w) sample, and pick the one that best
    balances goal progress, obstacle clearance, heading, and speed.
    
    Localization: prefers map-frame pose via TF2/AMCL, defaults to odom if no TF works.
    Goals come in through RViz's 2D Goal Pose tool on /{namespace}/goal_pose.
    """
    
    
    def __init__(self):
        # Read namespace early so we can remap TF topics at node init time.
        # ROS2 param override (--ros-args -p namespace:=/foo) still works;
        # the cli_args remaps just set up the TF subscriptions.
        import sys
        _ns = '/don'
        for i, arg in enumerate(sys.argv):
            if arg.startswith('namespace:='):
                _ns = arg.split(':=', 1)[1]
            elif arg == '-p' and i + 1 < len(sys.argv) and sys.argv[i+1].startswith('namespace:='):
                _ns = sys.argv[i+1].split(':=', 1)[1]

        super().__init__('dynamic_window_approach', cli_args=[
            '--ros-args',
            '-r', f'/tf:={_ns}/tf',
            '-r', f'/tf_static:={_ns}/tf_static',
        ])

#relevant hyperparams,,,, can edit here or test diff with command line
#twist
        self.declare_parameter('max_velocity', 0.4)
        self.declare_parameter('min_velocity', 0.0)
        self.declare_parameter('max_angular_velocity', 1.8)
        self.declare_parameter('min_angular_velocity', -1.8)
        self.declare_parameter('max_linear_acceleration', 0.5)
        self.declare_parameter('max_angular_acceleration', 1.5)
        self.declare_parameter('v_samples', 10) 
        self.declare_parameter('w_samples', 20)
        self.declare_parameter('lidar_angle_offset', 1.5708) # for whatever reason, the real life turtlebot needs to be shifted 90 degrees

        self.max_v = self.get_parameter('max_velocity').value
        self.min_v = self.get_parameter('min_velocity').value
        self.max_w = self.get_parameter('max_angular_velocity').value
        self.min_w = self.get_parameter('min_angular_velocity').value
        self.v_accel = self.get_parameter('max_linear_acceleration').value
        self.w_accel = self.get_parameter('max_angular_acceleration').value
        self.v_samples = self.get_parameter('v_samples').value
        self.w_samples = self.get_parameter('w_samples').value
        self.lidar_offset = self.get_parameter('lidar_angle_offset').value

        
# planning
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('prediction_steps', 50)
        self.declare_parameter('window_steps', 8)
        self.declare_parameter('LIDAR_downsample', 1)

        self.dt = self.get_parameter('dt').value
        self.steps = self.get_parameter('prediction_steps').value
        self.window_steps = self.get_parameter('window_steps').value
        self.LIDAR_downsample = self.get_parameter('LIDAR_downsample').value
        
# bubbles
        self.declare_parameter('critical_radius', 0.18)
        self.declare_parameter('safe_distance', 0.60)
        self.declare_parameter('emergency_stop_distance', 0.17)
        self.declare_parameter('max_lidar_range', 8.0)

        self.critical_radius = self.get_parameter('critical_radius').value
        self.safe_dist = self.get_parameter('safe_distance').value
        self.emergency_stop_dist = self.get_parameter('emergency_stop_distance').value
        self.max_lidar_range = self.get_parameter('max_lidar_range').value
        
# cost weights
        self.declare_parameter('weights.goal', 0.5)
        self.declare_parameter('weights.heading', 0.05)
        self.declare_parameter('weights.velocity', 0.2)
        self.declare_parameter('weights.smoothness', 0.05)
        self.declare_parameter('weights.obstacle', 0.1)

        self.w_goal = self.get_parameter('weights.goal').value
        self.w_heading = self.get_parameter('weights.heading').value
        self.w_velocity = self.get_parameter('weights.velocity').value
        self.w_smoothness = self.get_parameter('weights.smoothness').value
        self.w_obstacle = self.get_parameter('weights.obstacle').value
        
# recovery
        self.declare_parameter('recovery.linear_velocity', 0.0)
        self.declare_parameter('recovery.angular_velocity', 0.5)

        self.recovery_v = self.get_parameter('recovery.linear_velocity').value
        self.recovery_w = self.get_parameter('recovery.angular_velocity').value
        
#trajectory lines
        self.declare_parameter('visualize_trajectories', True)
        self.declare_parameter('trajectory_visualization_downsample', 10)

        self.visualize_trajectories = self.get_parameter('visualize_trajectories').value
        self.traj_downsample = self.get_parameter('trajectory_visualization_downsample').value
        
        self.declare_parameter('goal_tolerance', 0.3)
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        self.timer = self.create_timer(self.dt, self.nav_loop)
        self.goal = None
        self.emergency_scan_msg = None
        self.scan_msg = None
        self.odom_msg = None

        # Track which frame is actively used for pose/obstacles
        # Updated each nav_loop tick based on TF2 availability
        self.active_frame = 'odom'
        
#subs
        self.declare_parameter('namespace', '/don') #currently blank, but to get it to work on the robot, you need to use the robot name, for us it is /don
        self.namespace = self.get_parameter('namespace').value
        print(f"{self.namespace}/scan")

        self.emergencyLidar = self.create_subscription(LaserScan, f"{self.namespace}/scan", self.emergency_lidar_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, f"{self.namespace}/goal_pose", self.goal_callback, 10)
        self.odom_sub = Subscriber(self, Odometry, f'{self.namespace}/odom', qos_profile=qos_profile_sensor_data)
        self.scan_sub = Subscriber(self, LaserScan, f'{self.namespace}/scan', qos_profile=qos_profile_sensor_data)

#pubs
        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.twist_publish = self.create_publisher(TwistStamped, f'{self.namespace}/cmd_vel', qos_policy)
        self.debug_pub = self.create_publisher(Marker, f'{self.namespace}/debug_obstacles', 10)
        self.traj_pub = self.create_publisher(Marker, f'{self.namespace}/dwa/trajectories', 10)
        self.best_traj_pub = self.create_publisher(Marker, f'{self.namespace}/dwa/best_trajectory', 10)

#syncing
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self) 
        self.sync = ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.1)
        self.sync.registerCallback(self.synchronized_callback)
        


    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"New Goal Set: {self.goal}")

    def emergency_lidar_callback(self, msg: LaserScan):
        self.emergency_scan_msg = msg        
    def synchronized_callback(self, odom_msg, scan_msg):
        self.scan_msg = scan_msg
        self.odom_msg = odom_msg

# helper funcs

    def quat_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


    def get_robot_pose_map_frame(self):
        """
        Get robot pose from TF2 map->base_link transform.
        Returns (x, y, theta) in map frame, or None if transform unavailable.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = self.quat_to_yaw(q)
            return (x, y, theta)
        except Exception as e:
            return None


    def publish_trajectory_markers(self, trajectories, scores, best_idx):
        """
        Publishes the possible paths DWA can take, from red to green based on score, with chosen path in blue

        Args:
            trajectories (list of all trajectories): see predict_trajectories
            scores (list of all scores (matches trajectories size)): scores for all trajectories
            best_idx (int): index of the highest scoring path
        """
        if not self.visualize_trajectories:
            return
        
        now = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.active_frame
        marker.header.stamp = now
        marker.ns = "candidate_trajectories"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.pose.orientation.w = 1.0

        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            return
        
        smin = np.min(valid_scores)
        smax = np.max(valid_scores)
        srange = smax - smin if smax > smin else 1.0
        
        for i in range(0, len(trajectories), self.traj_downsample):
            if i == best_idx:
                continue
            
            traj = trajectories[i]
            score = scores[i]
            
            
            if np.isfinite(score):
                norm = (score - smin) / srange
                if norm < 0.5:
                    r, g, b = 1.0, norm * 2.0, 0.0
                else:
                    r, g, b = 1.0 - (norm - 0.5) * 2.0, 1.0, 0.0
                a = 0.3 + 0.4 * norm
            else:
                r, g, b, a = 0.5, 0.5, 0.5, 0.1 #gray for paths that go through objects
            

            for j in range(len(traj) - 1):
                p1 = Point()
                p1.x, p1.y, p1.z = float(traj[j, 0]), float(traj[j, 1]), 0.05
                p2 = Point()
                p2.x, p2.y, p2.z = float(traj[j+1, 0]), float(traj[j+1, 1]), 0.05
                marker.points.append(p1)
                marker.points.append(p2)
                
                c = ColorRGBA()
                c.r, c.g, c.b, c.a = r, g, b, a
                marker.colors.append(c)
                marker.colors.append(c)
        
        self.traj_pub.publish(marker)
        
        best_marker = Marker()
        best_marker.header.frame_id = self.active_frame
        best_marker.header.stamp = now
        best_marker.ns = "best_trajectory"
        best_marker.id = 1
        best_marker.type = Marker.LINE_STRIP
        best_marker.action = Marker.ADD
        best_marker.scale.x = 0.05
        best_marker.color.r, best_marker.color.g = 0.0, 1.0
        best_marker.color.b, best_marker.color.a = 1.0, 1.0
        best_marker.pose.orientation.w = 1.0
        
        for pt in trajectories[best_idx]:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
            best_marker.points.append(p)
        
        self.best_traj_pub.publish(best_marker)





    
#funcs
    def get_obstacles(self):
        """Convert the latest lidar scan from polar (range, angle) into world-frame
        XY obstacle points, using the robot's current pose for the transform"""
        if self.scan_msg is None:
            return np.array([])
        
        map_pose = self.get_robot_pose_map_frame()
        if map_pose is not None:
            x, y, theta = map_pose
        elif self.odom_msg is not None:
            theta = self.quat_to_yaw(self.odom_msg.pose.pose.orientation)
            x = self.odom_msg.pose.pose.position.x
            y = self.odom_msg.pose.pose.position.y
        else:
            return np.array([])

        
        #TODO May have to downsample for Turtlebot performance, LIDAR_downsample currently set to 1
        ranges = np.array(self.scan_msg.ranges)[::self.LIDAR_downsample]
        angles = np.linspace(
            self.scan_msg.angle_min,
            self.scan_msg.angle_max,
            len(self.scan_msg.ranges)
        )[::self.LIDAR_downsample]
        mask = (
            np.isfinite(ranges) &
            (ranges > 0.05) &
            (ranges > self.scan_msg.range_min) & 
            (ranges < self.scan_msg.range_max) & 
            (ranges < self.max_lidar_range)
        )
        valid_ranges = ranges[mask]
        valid_angles = angles[mask]

        world_angles = theta + valid_angles + self.lidar_offset        
        o_x = x + valid_ranges * np.cos(world_angles)
        o_y = y + valid_ranges * np.sin(world_angles)
        
        return np.column_stack((o_x, o_y))
        


    def cost_function(self, trajectories, final_vs, final_ws, obstacles, curr_x, curr_y):
        """Score each candidate trajectory with a weighted sum of:
          - goal:       how much closer the trajectory endpoint gets to the goal
          - heading:    alignment between final heading and direction to goal
          - obstacle:   clearance from nearest obstacle (via distanceGrid)
          - velocity:   preference for higher forward speed
          - smoothness: penalty for high angular velocity
        
        Trajectories whose closest obstacle is within critical_radius get
        hard-rejected (score = -inf)."""
        
        final_pos = trajectories[:, -1, :2]
        curr_dist = np.linalg.norm(self.goal - np.array([curr_x, curr_y]))
        goal_dists = np.linalg.norm(self.goal - final_pos, axis=1)
        
        #goal score
        max_prog = self.max_v * self.dt * self.steps
        progress = curr_dist - goal_dists
        g_score = np.clip((progress + max_prog) / (2 * max_prog), 0.0, 1.0)
        
        #heading score
        dx = self.goal[0] - final_pos[:, 0]
        dy = self.goal[1] - final_pos[:, 1]
        goal_theta = np.arctan2(dy, dx)
        final_theta = trajectories[:, -1, 2]
        heading_err = np.abs(np.arctan2(
            np.sin(goal_theta - final_theta),
            np.cos(goal_theta - final_theta)
        ))
        h_score = 1.0 - (heading_err / np.pi)
        
        #obstacle score — min_dists used for hard-rejection below
        if len(obstacles) > 0:
            all_pts = trajectories[:, :, :2]
            diffs = obstacles[None, None, :, :] - all_pts[:, :, None, :]
            dists = np.linalg.norm(diffs, axis=3)
            min_dists = np.min(dists, axis=(1, 2))
        else:
            min_dists = np.full(len(final_vs), self.max_lidar_range)

        o_score_list = dg.get_all_path_costs(trajectories, obstacles)
        o_score = np.array(o_score_list)
        
        #velocity and smoothness
        v_score = np.clip(final_vs / self.max_v, 0.0, 1.0)
        w_score = np.clip(1.0 - (np.abs(final_ws) / self.max_w), 0.0, 1.0)
        
        #combine
        scores = (
            self.w_goal * g_score +
            self.w_heading * h_score +
            self.w_velocity * v_score +
            self.w_smoothness * w_score +
            self.w_obstacle * o_score
        )
        
        #hard reject if too close to objects
        crit = np.where(np.abs(final_vs) > 0.1, self.critical_radius, self.emergency_stop_dist + 0.01)
        scores[min_dists < crit] = -np.inf
        
        return scores, min_dists
    
        
    # changed to do vector mult,, similar logic but applies to all v w 
    def predict_trajectories(self, v_arr, w_arr, curr_x, curr_y, curr_theta, steps=30, dt=0.1):
        """Forward-simulate all (v, w) pairs in parallel using vectorized cumulative sums.
        Each pair is held constant for `steps` timesteps of length `dt`.
        Returns (trajectories, final_vs, final_ws) where trajectories is (N, steps, 3)
        with columns [x, y, theta]."""
        
                
        #copy over v and w across array
        vs = np.tile(v_arr[:, None], (1, steps))
        ws = np.tile(w_arr[:, None], (1, steps))
        
        thetas = np.cumsum(ws * dt, axis=1) + curr_theta
        xs = np.cumsum(vs * np.cos(thetas) * dt, axis=1) + curr_x
        ys = np.cumsum(vs * np.sin(thetas) * dt, axis=1) + curr_y
        
        trajectories = np.stack((xs, ys, thetas), axis=2)
        return trajectories, vs[:, -1], ws[:, -1]
    
    
        
    def dynamic_window(self, curr_v, curr_w):
        """Compute the reachable velocity bounds given current (v, w) and
        acceleration limits over the next window_steps * dt seconds.
        Returns (v_max, v_min, w_max, w_min) clamped to global limits."""

        window_time = self.dt * self.window_steps #look further ahead

        v_range = self.v_accel * window_time
        w_range = self.w_accel * window_time

        poss_v_max = min(curr_v + v_range, self.max_v)
        poss_v_min = max(curr_v - v_range, self.min_v)
        poss_w_max = min(curr_w + w_range, self.max_w)
        poss_w_min = max(curr_w - w_range, self.min_w)

        return poss_v_max, poss_v_min, poss_w_max, poss_w_min


    def nav_loop(self):
        """Main control loop, called every dt seconds. Pipeline:
        1. Emergency stop check (front-cone lidar)
        2. Get pose (map frame if AMCL available, else odom)
        3. Goal-reached check
        4. Compute dynamic window → sample (v,w) → simulate trajectories
        5. Score trajectories → pick best → publish cmd_vel
        Falls back to recovery behavior if all trajectories are rejected."""
        if self.goal is None:
            return 
        if self.odom_msg is None or self.scan_msg is None:
            return
        
        if self.emergency_scan_msg is not None:
            ranges = np.array(self.emergency_scan_msg.ranges)            
            ranges = np.where(
                np.isfinite(ranges) & (ranges > 0.05),
                ranges,
                10.0
            )
            n = len(ranges)
            angle_range = self.emergency_scan_msg.angle_max - self.emergency_scan_msg.angle_min
            front_idx = int((-self.lidar_offset - self.emergency_scan_msg.angle_min) / angle_range * n) % n
            cone_width = int(n / 8)
            indices = np.arange(front_idx - cone_width, front_idx + cone_width) % n
            front_ranges = ranges[indices]
    
            if np.min(front_ranges) < self.emergency_stop_dist:
                self.get_logger().warn("EMERGENCY STOP: Obstacle too close!")
                stop_cmd = TwistStamped()
                stop_cmd.header.stamp = self.get_clock().now().to_msg()
                stop_cmd.header.frame_id = 'base_link'
                stop_cmd.twist.linear.x = 0.0
                stop_cmd.twist.angular.z = 0.0
                self.twist_publish.publish(stop_cmd)
                return
            

        # get pose (try map frame, else odom)
        map_pose = self.get_robot_pose_map_frame()
        if map_pose is not None:
            curr_x, curr_y, curr_theta = map_pose
            self.active_frame = 'map'
        else:
            curr_x = self.odom_msg.pose.pose.position.x 
            curr_y = self.odom_msg.pose.pose.position.y
            curr_theta = self.quat_to_yaw(self.odom_msg.pose.pose.orientation)
            self.active_frame = 'odom'
        curr_v = self.odom_msg.twist.twist.linear.x
        curr_w = self.odom_msg.twist.twist.angular.z


        dist = np.linalg.norm(self.goal - np.array([curr_x, curr_y]))
        if dist < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            stop_cmd = TwistStamped()
            stop_cmd.header.stamp = self.get_clock().now().to_msg()
            stop_cmd.header.frame_id = 'base_link'
            stop_cmd.twist.linear.x = 0.0
            stop_cmd.twist.angular.z = 0.0
            self.twist_publish.publish(stop_cmd)
            self.goal = None
            return   


        poss_v_max, poss_v_min, poss_w_max, poss_w_min = self.dynamic_window(curr_v, curr_w)
        poss_v = np.linspace(poss_v_min, poss_v_max, self.v_samples)  
        poss_w = np.linspace(poss_w_min, poss_w_max, self.w_samples)
        v_arr, w_arr = np.meshgrid(poss_v, poss_w)
        v_arr = v_arr.flatten()
        w_arr = w_arr.flatten()
        trajectories, final_vs, final_ws = self.predict_trajectories(
            v_arr, w_arr, curr_x, curr_y, curr_theta, self.steps, self.dt
        )


        #view what DWA sees as obstacles in RVIZ (should also be adjusted to map frame now)
        obstacles = self.get_obstacles()
        if len(obstacles) > 0:
            marker = Marker()
            marker.header.frame_id = self.active_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            for obs in obstacles:
                p = Point()
                p.x = float(obs[0])
                p.y = float(obs[1])
                p.z = 0.0
                marker.points.append(p)
            self.debug_pub.publish(marker)

        scores, min_dists = self.cost_function(trajectories, final_vs, final_ws, obstacles, curr_x, curr_y)
        best_idx = np.argmax(scores)
        self.publish_trajectory_markers(trajectories, scores, best_idx)
            
        if scores[best_idx] == -np.inf:
            self.get_logger().warn("Too close backing up")
            best_v = self.recovery_v
            best_w = self.recovery_w
        else:
            best_v = float(final_vs[best_idx])
            best_w = float(final_ws[best_idx])


        self.message = TwistStamped()
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.message.header.frame_id = 'base_link'
        self.message.twist.linear.x = best_v
        self.message.twist.angular.z = best_w
        self.twist_publish.publish(self.message)



def main(args=None):
    rclpy.init(args=args)
    node = DWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()