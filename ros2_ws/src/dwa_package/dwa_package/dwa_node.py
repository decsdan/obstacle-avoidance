import rclpy
import numpy as np
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
#NOTE: For callbacks, we wanna do as little as possible, just save messages and exit, and process the messages in another function, they block all other callbacks when being processed

class DWA(Node):
    """_summary_

    Args:
        Node (_type_): _description_
    """    
    def __init__(self):
        """ subscribes to lidar (LaserScan) and Odometry, and preps to publish to movement (Twist), then runs nav_loop over and over
        """
        super().__init__('dynamic_window_approach')   

        self.callback_group = ReentrantCallbackGroup() #adding a callback group for multi threaded work
        self.dt = 0.1 #how long each step is in time
        #relevant hyperprams for the DWA functions, I think these are more realistic
        self.dt = 0.1
        self.max_v = 0.5 
        self.min_v = -0.1
        self.max_w = 1.5
        self.min_w = -1.5
        self.v_accel = 0.5
        self.w_accel = 0.5
        self.steps = 10 #how far to look ahead
        self.new_data = False

        self.emergencyLidar = self.create_subscription(LaserScan, '/scan', self.emergency_lidar_callback,10) #the 10 here is how many messages it keeps in queue, not sure what a good number would be but demo_node had 10
        #self.odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #these are regular subscribers, so things that arent filtered by Approximate TimeSync (we can use these for emergency override stuff, e.g. LIDAR detects an object extremely close but odom hasnt published yet, so we cant trust DWA to stop)
        self.goal = None
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.timer = self.create_timer(self.dt, self.nav_loop, callback_group=self.callback_group) #edited to reference the callback group
        
        #added this to match reliability policy of robot
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.twist_publish = self.create_publisher(TwistStamped, '/cmd_vel', qos_policy)

        self.sync = ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.5) # 0.5s tolerance
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info('Synchronizer node started.')

        self.emergency_scan_msg = None
        self.scan_msg = None
        self.odom_msg = None


        self.odom_count = 0
        self.scan_count = 0


        


    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"New Goal Set: {self.goal}")

    def emergency_lidar_callback(self, msg: LaserScan):
        """gets unsynced raw lidar from the LaserScan directly, for use in emergency stop
        from https://docs.ros.org/en/humble/p/sensor_msgs/

        """
        self.emergency_scan_msg = msg
        self.scan_count += 1
        #self.get_logger().info(f"Recieved scan message {self.scan_count}") #also if you want to see the scan messages, just scan_msg instead of scan_count


    #commented out basic odom_callback, we can decide to use this if we want but for now we have synced messages
    '''
    def odom_callback(self, msg: Odometry):
        """gets odometry from Odometry

https://docs.ros.org/en/humble/p/nav_msgs/msg/Odometry.html
        '''

    def synchronized_callback(self, odom_msg, scan_msg):
        # self.get_logger().info(f"Synced: Odom time {odom_msg.header.stamp}, Scan time {scan_msg.header.stamp}")
        self.scan_msg = scan_msg
        self.odom_msg = odom_msg
        self.new_data = True




    def cost_function(self,trajectory): #TODO implement obstacle avoidance into cost function, not just final state function

        #self.get_logger().info(f"TESTING COST FUNCTION")
        #get final location, twist, direction
        x = trajectory[-1][0]
        y = trajectory[-1][1]
        theta= trajectory[-1][2]
        w = trajectory[-1][3]
        v = trajectory[-1][4]

        final_pos = [x,y]
        dist = np.linalg.norm(self.goal - final_pos)
        score = -dist + 0.1 * v -0.1 * abs(w)

        return score
    
    def predict_trajectory(self,v,w, curr_x, curr_y, curr_theta,steps=30, dt=0.1):

        trajectory = np.zeros((steps,5))

        for i in range(steps):
            curr_x += v * np.cos(curr_theta) *dt
            curr_y += v * np.sin(curr_theta) *dt
            curr_theta += w *dt
            trajectory[i] = [curr_x,curr_y,curr_theta,v,w]

        return trajectory
    
    def dynamic_window(self, curr_v, curr_w):
        poss_v_max = curr_v + self.v_accel * self.dt 
        poss_v_max = min(poss_v_max,self.max_v) 

        poss_v_min = curr_v - self.v_accel* self.dt
        poss_v_min = max(poss_v_min, self.min_v)

        poss_w_max = curr_w + self.w_accel * self.dt
        poss_w_max = min(poss_w_max, self.max_w)

        poss_w_min = curr_w - self.w_accel* self.dt
        poss_w_min = max(poss_w_min, self.min_w)

        return poss_v_max, poss_v_min, poss_w_max, poss_w_min


    # the orientation needs to be converted from 3d (quaternion) to 2d (yaw)
    def quat_to_yaw(self, q):
        """
        Convert a quaternion into euler angle yaw
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def nav_loop(self):
        """_summary_
        """   
        if self.goal is None: #need to set a goal via Rviz2
            return 
        if self.new_data is False:
            return  


        curr_x = self.odom_msg.pose.pose.position.x #get pose and twist
        curr_y = self.odom_msg.pose.pose.position.y
        curr_theta = self.quat_to_yaw(self.odom_msg.pose.pose.orientation)
        curr_v = self.odom_msg.twist.twist.linear.x
        curr_w = self.odom_msg.twist.twist.angular.z

        dist = np.linalg.norm(self.goal - np.array([curr_x, curr_y])) #created goal reached catch
        if dist < 0.3: 
            self.get_logger().info("goal hit")
            stop_cmd = Twist()
            self.twist_publish.publish(stop_cmd)
            self.new_data = False
            return
        best_score = -np.inf       

        best_v = 0.0
        best_w = 0.0
        poss_v_max, poss_v_min, poss_w_max, poss_w_min = self.dynamic_window(curr_v, curr_w)

        poss_v = np.linspace(poss_v_min, poss_v_max,5)  #should test cost function 25 times (within max/min velocity)
        poss_w = np.linspace(poss_w_min, poss_w_max, 5)
        for v in poss_v:
            for w in poss_w:
                trajectory = self.predict_trajectory(v,w, curr_x, curr_y, curr_theta,steps=self.steps, dt=self.dt)
                score = self.cost_function(trajectory)
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        
        self.get_logger().info(f"Best V and W set! {best_v}, {best_w}")
        
        #added twistStamped here, turtlebot4 requires twiststamped
        self.message = TwistStamped()
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.message.header.frame_id='base_link'
        self.message.twist.linear.x = float(best_v)
        self.message.twist.angular.z = float(best_w)
        self.twist_publish.publish(self.message)












#edited to handle calcs and twist publishing to not stop and start
def main(args=None):
    rclpy.init(args=args)
    node = DWA()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



