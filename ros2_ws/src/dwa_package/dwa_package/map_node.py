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
        self.map_msg = None
        self.get_logger().info("Distance grid node started")

    def map_callback(self, msg: OccupancyGrid):
        """
        This function is referenced in the subscription to the /map topic. Basically every time
        the /map topic publishes data, this function will be automatically called. This should call
        update_distance_grid() and then publish the distance grid.

        Args:
            - OccupancyGrid: the new data from the map
        """
        self.map_msg = msg

        # Convert OccupancyGrid toi numpy (row-major)
        self.map_data = np.array(
            msg.data, dtype=np.int8
        ).reshape((msg.info.height, msg.info.width))

        self.get_logger().info(
            f"Received map {msg.info.width}x{msg.info.height}"
        )

        self.update_distance_grid()
        self.publish_map()

    def update_distance_grid(self):
        if self.map_data is None:
            return

        OCCUPIED = 100

        height, width = self.map_data.shape

        # distanceGrid expects (width, height)
        self.distance_grid = dg.getDistanceGrid(
            self.map_data,
            width,
            height,
            OCCUPIED
        )

    def publish_map(self):
        if self.distance_grid is None or self.map_msg is None:
            return

        msg = OccupancyGrid()

        # original info for the grid
        msg.header = self.map_msg.header
        msg.info = self.map_msg.info

        # scaling every distance so its between 1 and 100 (or -1 for unkown) so rviz works
        max_dist = np.max(self.distance_grid)
        if max_dist > 0:
            scaled = (self.distance_grid / max_dist) * 100
        else:
            scaled = self.distance_grid.copy()

        # Preserve unknown cells
        scaled[self.map_data == -1] = -1

        msg.data = scaled.astype(np.int8).flatten().tolist()

        self.publisher.publish(msg)
        #this message totally spams the terminal
        #self.get_logger().info("Published /distance_map")
        self.get_logger().debug("Published /distance_map")


def main(args = None):
    rclpy.init(args=args)
    node = MapListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
