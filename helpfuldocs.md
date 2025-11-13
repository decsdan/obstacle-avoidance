# Helpful Commands/Links in ROS2 documentation

## Used Within DWA

#### Odometry:

https://docs.ros.org/en/humble/p/nav_msgs/msg/Odometry.html

from nav_msgs.msg import Odometry


##### TwistWithCovariance:

https://docs.ros.org/en/humble/p/geometry_msgs/msg/TwistWithCovariance.html

Included as a value "twist" within odometry message

This is not the same as publishing to Twist (telling the robot to move, located below), so we don't have to call from geometry_msgs.msg import Twist to get the twist vals here

##### PoseWithCovariance:

https://docs.ros.org/en/humble/p/geometry_msgs/msg/PoseWithCovariance.html

Included as value "pose" within odometry message



#### LaserScan:

https://docs.ros.org/en/humble/p/sensor_msgs/msg/LaserScan.html


from sensor_msgs.msg import LaserScan


#### Twist (publishing)

https://docs.ros.org/en/humble/p/geometry_msgs/msg/Twist.html

from geometry_msgs.msg import Twist



#### Ensuring Time Sync

Odometry posts like once every 30 LIDAR posts, meaning if we are pulling from the raw subscription queues, the LIDAR data will often be from earlier than the Odometry (a mismatch),
so we need to use message_filters to deal with it 

https://docs.ros.org/en/humble/p/message_filters/

https://docs.ros.org/en/humble/p/message_filters/doc/Tutorials/Approximate-Synchronizer-Python.html

from message_filters import Subscriber, ApproximateTimeSynchronizer
