import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dennisd2/obstacle-avoidance-comps/ros2_ws/install/demo_package'
