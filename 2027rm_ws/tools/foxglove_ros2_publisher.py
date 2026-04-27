#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker


def quat_to_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def matrix_to_quat(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return (
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
            0.25 / s,
        )
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        return (
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    if m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        return (
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
    return (
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
        (m[1][0] - m[0][1]) / s,
    )


def matmul3(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def cv_camera_pose_to_visual_pose(tvec, quat):
    # OpenCV optical: X right, Y down, Z forward.
    # ROS/Foxglove view: X forward, Y left, Z up.
    cv_to_visual = (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    position = (
        float(tvec[2]),
        -float(tvec[0]),
        -float(tvec[1]),
    )
    rotation = matmul3(cv_to_visual, quat_to_matrix(tuple(float(v) for v in quat)))
    return position, matrix_to_quat(rotation)


class DebugImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__('rm_debug_image_publisher')
        self.frame_path = Path(os.getenv('RM_DEBUG_FRAME_PATH', '/tmp/rm_rerun/latest.jpg'))
        self.pnp_path = Path(os.getenv('RM_DEBUG_PNP_PATH', '/tmp/rm_rerun/pnp_latest.json'))
        self.topic = os.getenv('RM_DEBUG_TOPIC', '/debug/image/compressed')
        self.pub = self.create_publisher(CompressedImage, self.topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/debug/pnp/pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/debug/pnp/armor', 10)
        self.last_mtime_ns = -1
        self.last_pnp_mtime_ns = -1
        self.sent = 0
        self.create_timer(0.02, self._tick)
        self.get_logger().info(
            f'Publishing {self.frame_path} -> {self.topic}, {self.pnp_path} -> /debug/pnp/pose + /debug/pnp/armor'
        )

    def _tick(self) -> None:
        self._publish_image()
        self._publish_pnp()

    def _publish_image(self) -> None:
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
            self.get_logger().info(f'published {self.sent} image frames')

    def _publish_pnp(self) -> None:
        if not self.pnp_path.exists():
            return

        stat = self.pnp_path.stat()
        mtime_ns = stat.st_mtime_ns
        if mtime_ns == self.last_pnp_mtime_ns:
            return

        try:
            data = json.loads(self.pnp_path.read_text())
        except Exception as exc:
            self.get_logger().warning(f'failed to read pnp json: {exc}')
            return

        if not data.get('has_pose', False):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'camera_link'
            marker.ns = 'armor'
            marker.id = 1
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
            self.last_pnp_mtime_ns = mtime_ns
            return

        stamp = self.get_clock().now().to_msg()
        position, orientation = cv_camera_pose_to_visual_pose(
            data['tvec_m'], data['quat_xyzw']
        )

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'camera_link'
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        self.pose_pub.publish(pose)

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = 'camera_link'
        marker.ns = 'armor'
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.01
        marker.scale.y = float(data['armor_size_m'][0])
        marker.scale.z = float(data['armor_size_m'][1])
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.9
        marker.color.a = 0.7
        self.marker_pub.publish(marker)

        self.last_pnp_mtime_ns = mtime_ns


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
