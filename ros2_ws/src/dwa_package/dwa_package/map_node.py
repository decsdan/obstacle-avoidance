"""
Ideally this node will handle the creation and updating of the distance grid.
This will make the code more modular and the main dwa_node will be able to
just subscribe and only deal with the distance grid when incorperating into
the cost function.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import distanceGrid as dg

class MapListener(Node):
    def __init__(self):
        super().__init__('map_listener')
        self.subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.publisher = self.create_publisher(OccupancyGrid, '/distance_map', 10)
        self.map_data = None
        self.distance_grid = None

    def map_callback(self, msg: OccupancyGrid):
        """
        This function is referenced in the subscription to the /map topic. Basically every time
        the /map topic publishes data, this function will be automatically called. This should call
        update_distance_grid() and then publish the distance grid.

        Args:
            - OccupancyGrid: the new data from the map
        """
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.get_logger().info(f"Received map: {msg.info.width}x{msg.info.height}")
        # Example: print the value of the top-left cell
        self.get_logger().info(f"Top-left cell value: {self.map_data[0,0]}")

    def update_distance_grid(self):
        #TODO
        pass

    def publish_map(self):
        #TODO: check type referenced when instantiating the publisher before done
        msg = self.distance_grid
        self.publisher.publish(msg)
        self.get_logger().info("Published map to /self_map")

def main(args = None):
    rclpy.init(args=args)
    node = MapListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
