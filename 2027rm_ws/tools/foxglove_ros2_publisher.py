#!/usr/bin/env python3
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class DebugImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__('rm_debug_image_publisher')
        self.frame_path = Path(os.getenv('RM_DEBUG_FRAME_PATH', '/tmp/rm_rerun/latest.jpg'))
        self.topic = os.getenv('RM_DEBUG_TOPIC', '/debug/image/compressed')
        self.pub = self.create_publisher(CompressedImage, self.topic, 10)
        self.last_mtime_ns = -1
        self.sent = 0
        self.create_timer(0.02, self._tick)
        self.get_logger().info(f'Publishing {self.frame_path} -> {self.topic}')

    def _tick(self) -> None:
        if not self.frame_path.exists():
            return

        stat = self.frame_path.stat()
        mtime_ns = stat.st_mtime_ns
        if mtime_ns == self.last_mtime_ns:
            return

        payload = self.frame_path.read_bytes()
        if not payload:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = payload
        self.pub.publish(msg)

        self.last_mtime_ns = mtime_ns
        self.sent += 1
        if self.sent % 60 == 0:
            self.get_logger().info(f'published {self.sent} frames')


def main() -> None:
    rclpy.init()
    node = DebugImagePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
