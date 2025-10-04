#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/haowen/yolov5_env/lib/python3.10/site-packages')  # 确保当前目录在路径中
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import math

class BallPositionSubscriber(Node):
    def __init__(self):
        super().__init__('ball_position_subscriber')
        self.subscription = self.create_subscription(
            PointStamped,
            'ball_position',
            self.position_callback,
            10)
        self.get_logger().info("Ball position subscriber started")

    def position_callback(self, msg):
        if not self.is_nan(msg.point):
            self.get_logger().info(
                f"Received ball position: X={msg.point.x:.2f}mm, "
                f"Y={msg.point.y:.2f}mm, Z={msg.point.z:.2f}mm"
            )
        else:
            self.get_logger().info("Received: No ball detected")

    def is_nan(self, point):
        return math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z)

def main(args=None):
    rclpy.init(args=args)
    subscriber = BallPositionSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import math

class BallPositionSubscriber(Node):
    def __init__(self):
        super().__init__('ball_position_subscriber')
        self.subscription = self.create_subscription(
            PointStamped,
            'ball_position',
            self.position_callback,
            10)
        self.get_logger().info("Ball position subscriber started")

    def position_callback(self, msg):
        if not self.is_nan(msg.point):
            self.get_logger().info(
                f"Received ball position: X={msg.point.x:.2f}mm, "
                f"Y={msg.point.y:.2f}mm, Z={msg.point.z:.2f}mm"
            )
        else:
            self.get_logger().info("Received: No ball detected")

    def is_nan(self, point):
        return math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z)

def main(args=None):
    rclpy.init(args=args)
    subscriber = BallPositionSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()