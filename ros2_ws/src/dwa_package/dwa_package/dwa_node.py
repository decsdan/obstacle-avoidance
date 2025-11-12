import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer

#NOTE: For callbacks, we wanna do as little as possible, just save messages and exit, and process the messages in another function, they block all other callbacks when being processed

class DWA(Node):
    """_summary_

    Args:
        Node (_type_): _description_
    """    
    def __init__(self):
        """ subscribes to lidar (LaserScan) and Odometry, and preps to publish to movement (Twist), then runs nav_loop over and over
        """
        dt = 0.1
        self.new_data = False
        super().__init__('dynamic_window_approach')        
        self.emergencyLidar = self.create_subscription(LaserScan, '/scan', self.emergency_lidar_callback,10) #the 10 here is how many messages it keeps in queue, not sure what a good number would be but demo_node had 10
        #self.odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #these are regular subscribers, so things that arent filtered by Approximate TimeSync (we can use these for emergency override stuff, e.g. LIDAR detects an object extremely close but odom hasnt published yet, so we cant trust DWA to stop)
        self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.twist_publish = self.create_publisher(Twist, '/cmd_vel',10)
        self.timer = self.create_timer(dt, self.nav_loop) #how often navloop runs


        self.sync = ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.1) # 0.1s tolerance
        self.sync.registerCallback(self.synchronized_callback)
        self.get_logger().info('Synchronizer node started.')

        self.emergency_scan_msg = None
        self.scan_msg = None
        self.odom_msg = None


        self.odom_count = 0
        self.scan_count = 0



        #relevant hyperprams for the DWA functions
        self.dt = 0.1
        self.max_v = 5 
        self.min_v = -5
        self.max_w = 5
        self.min_w = -5
        self.v_accel = 1
        self.w_accel = 1



    def emergency_lidar_callback(self, msg: LaserScan):
        """gets unsynced raw lidar from the LaserScan directly, for use in emergency stop
        from https://docs.ros.org/en/humble/p/sensor_msgs/


        # Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

    std_msgs/Header header # timestamp in the header is the acquisition time of
                             # the first ray in the scan.
                             #
                             # in frame frame_id, angles are measured around
                             # the positive Z axis (counterclockwise, if Z is up)
                             # with zero angle being forward along the x axis

    float32 angle_min            # start angle of the scan [rad]
    float32 angle_max            # end angle of the scan [rad]
    float32 angle_increment      # angular distance between measurements [rad]

    float32 time_increment       # time between measurements [seconds] - if your scanner
                             # is moving, this will be used in interpolating position
                             # of 3d points
    float32 scan_time            # time between scans [seconds]

    float32 range_min            # minimum range value [m]
    float32 range_max            # maximum range value [m]

    float32[] ranges             # range data [m]
                             # (Note: values < range_min or > range_max should be discarded)
    float32[] intensities    # intensity data [device-specific units].  If your
                             # device does not provide intensities, please leave
                             # the array empty.
        """        
        self.emergency_scan_msg = msg
        self.scan_count += 1
        #self.get_logger().info(f"Recieved scan message {self.scan_count}") #also if you want to see the scan messages, just scan_msg instead of scan_count


    #commented out basic odom_callback, we can decide to use this if we want but for now we have synced messages
    '''
    def odom_callback(self, msg: Odometry):
        """gets odometry from Odometry

https://docs.ros.org/en/humble/p/nav_msgs/msg/Odometry.html

# This represents an estimate of a position and velocity in free space.
# The pose in this message should be specified in the coordinate frame given by header.frame_id
# The twist in this message should be specified in the coordinate frame given by the child_frame_id

# Includes the frame id of the pose parent.
    std_msgs/Header header

# Frame id the pose points to. The twist is in this coordinate frame.
    string child_frame_id

# Estimated pose that is typically relative to a fixed world frame.
    geometry_msgs/PoseWithCovariance pose

# Estimated linear and angular velocity relative to child_frame_id.
    geometry_msgs/TwistWithCovariance twist
            TWISTWITHCOVARIANCE  :


            # This expresses velocity in free space with uncertainty.

                Twist twist
                    # This expresses velocity in free space broken into its linear and angular parts.

                        Vector3  linear
                        Vector3  angular

            # Row-major representation of the 6x6 covariance matrix
            # The orientation parameters use a fixed-axis representation.
            # In order, the parameters are:
            # (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
                float64[36] covariance

        """        
        
        self.odom_msg = msg
        self.odom_count += 1
        #self.get_logger().info(f"Recieved odom message {self.odom_count}") #also if you want to see the odom messages, just odom_msg instead of odom_count

        '''

    def synchronized_callback(self, odom_msg, scan_msg):
        self.get_logger().info(f"Synced: Odom time {odom_msg.header.stamp}, Scan time {scan_msg.header.stamp}")
        self.scan_msg = scan_msg
        self.odom_msg = odom_msg
        self.new_data = True




    def cost_function(self,trajectory, goal): #TODO implement obstacle avoidance into cost function, not just final state function

        #self.get_logger().info(f"TESTING COST FUNCTION")
        #get final location, twist, direction
        x = trajectory[-1][0]
        y = trajectory[-1][1]
        theta= trajectory[-1][2] #currently this just is a distance from Goal cost function
        w = trajectory[-1][3]
        v = trajectory[-1][4]


        final_pos = [x,y]
        dist = np.linalg.norm(goal - final_pos)
        score = -dist + 0.1 * v -0.1 * abs(w)

        return score
    
    def predict_trajectory(self,v,w,steps=10, dt=0.1):
        x, y, theta = 1,2,3 #TODO, need to pull pose from odometry
        trajectory = np.zeros((steps,5))

        for i in range(steps):
            x += v * np.cos(theta) *dt
            y += v * np.sin(theta) *dt
            theta += w *dt
            trajectory[i] = [x,y,theta,v,w]

        return trajectory
    
    def dynamic_window(self):
        curr_v = 0 
        curr_w = 0

        poss_v_max = curr_v + self.v_accel * self.dt 
        poss_v_max = min(poss_v_max,self.max_v) 

        poss_v_min = curr_v - self.v_accel* self.dt
        poss_v_min = max(poss_v_min, self.min_v)

        poss_w_max = curr_w + self.w_accel * self.dt
        poss_w_max = min(poss_w_max, self.max_w)

        poss_w_min = curr_w - self.w_accel* self.dt
        poss_w_min = max(poss_w_min, self.min_w)

        return poss_v_max, poss_v_min, poss_w_max, poss_w_min



    def nav_loop(self):
        """_summary_
        """   

        self.get_logger().info(f"Running NAV LOOP")
        if self.new_data is False: #only run through nav loop fully if weve actually gotten new data, or any data
            return  
        #TODO implement proper vars here, were gonna have to reference odom_msg pose and twist, im pretty sure to reference twist we have to do odom_msg.twist.twist something something


        #DWA sorta code, need to figure out how to reference the vals from the published scans
        #TODO I think we need to create a node that publishes a goal being reached when we hit a certain val? unsure. Just coding in some stuff 


        goal = np.array([10.0,10.0])
        best_score = -np.inf
        best_v = 0.0
        best_w = 0.0
        poss_v_max, poss_v_min, poss_w_max, poss_w_min = self.dynamic_window()
        poss_v = np.linspace(poss_v_min, poss_v_max,5)  #should test cost function 25 times (within max/min velocity)
        poss_w = np.linspace(poss_w_min, poss_w_max, 5)
        for v in poss_v:
            for w in poss_w:
                trajectory = self.predict_trajectory(v,w)
                score = self.cost_function(trajectory,goal)
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        

        #TODO command robot to twist based on best v best w, and publish


        self.new_data =  False 














def main(args=None):
    rclpy.init(args=args)
    node = DWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



