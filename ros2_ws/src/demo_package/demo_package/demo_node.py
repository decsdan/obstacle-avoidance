import rospy
from geometry_msgs.msg import Twist

class control_tb3():
    def __init__(self):
        rospy.init_node('control_tb3', anonymous=True)
        rospy.loginfo("Press CTRL + C to terminate the TurtleBot3")
        rospy.on_shutdown(self.shutdown)
        
        self.cmd_vel_object = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 1
        self.vel_msg.angular.z = 0.5
        
        r = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            self.cmd_vel_object.publish(self.vel_msg)
            r.sleep()
            
    def shutdown(self):
        self.cmd_vel_object.publish(Twist())
        rospy.loginfo("The TB3 is stopping")
        rospy.sleep(1)
        
if __name__ == '__main__':
    try:
        control_tb3()
    except rospy.ROSInterruptException:
        pass
    except:
        rospy.loginfo("Turtlebot node is terminated")
