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


def cv_camera_point_to_visual(tvec):
    return (
        float(tvec[2]),
        -float(tvec[0]),
        -float(tvec[1]),
    )


def cv_yaw_pitch_to_visual_quat(yaw_rad, pitch_rad=0.0):
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    cp = math.cos(float(pitch_rad))
    sp = math.sin(float(pitch_rad))
    # Marker local axes: X=armor normal/thickness, Y=armor width, Z=armor height.
    # Keep width in the ground plane and height on visual Z, then tilt around local Y.
    yaw_rot = (
        (c, s, 0.0),
        (-s, c, 0.0),
        (0.0, 0.0, 1.0),
    )
    pitch_rot = (
        (cp, 0.0, sp),
        (0.0, 1.0, 0.0),
        (-sp, 0.0, cp),
    )
    visual_rot = matmul3(yaw_rot, pitch_rot)
    return matrix_to_quat(visual_rot)


class DebugImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__('rm_debug_image_publisher')
        self.frame_path = Path(os.getenv('RM_DEBUG_FRAME_PATH', '/tmp/rm_rerun/latest.jpg'))
        self.pnp_path = Path(os.getenv('RM_DEBUG_PNP_PATH', '/tmp/rm_rerun/pnp_latest.json'))
        self.rgo_path = Path(os.getenv('RM_DEBUG_RGO_PATH', '/tmp/rm_rerun/rgo_latest.json'))
        self.topic = os.getenv('RM_DEBUG_TOPIC', '/debug/image/compressed')
        self.pub = self.create_publisher(CompressedImage, self.topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/debug/pnp/pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/debug/pnp/armor', 10)
        self.rgo_center_pub = self.create_publisher(Marker, '/debug/rgo/center', 10)
        self.rgo_armor_pub = self.create_publisher(Marker, '/debug/rgo/armor', 10)
        self.last_mtime_ns = -1
        self.last_pnp_mtime_ns = -1
        self.last_rgo_mtime_ns = -1
        self.sent = 0
        self.create_timer(0.02, self._tick)
        self.get_logger().info(
            f'Publishing {self.frame_path} -> {self.topic}, {self.pnp_path} -> /debug/pnp/*, {self.rgo_path} -> /debug/rgo/*'
        )

    def _tick(self) -> None:
        self._publish_image()
        self._publish_pnp()
        self._publish_rgo()

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

    def _delete_rgo_markers(self, stamp) -> None:
        center = Marker()
        center.header.stamp = stamp
        center.header.frame_id = 'camera_link'
        center.ns = 'rgo_center'
        center.id = 1
        center.action = Marker.DELETE
        self.rgo_center_pub.publish(center)

        for idx in range(4):
            armor = Marker()
            armor.header.stamp = stamp
            armor.header.frame_id = 'camera_link'
            armor.ns = 'rgo_armor'
            armor.id = idx
            armor.action = Marker.DELETE
            self.rgo_armor_pub.publish(armor)

    def _publish_rgo(self) -> None:
        if not self.rgo_path.exists():
            return

        stat = self.rgo_path.stat()
        mtime_ns = stat.st_mtime_ns
        if mtime_ns == self.last_rgo_mtime_ns:
            return

        try:
            data = json.loads(self.rgo_path.read_text())
        except Exception as exc:
            self.get_logger().warning(f'failed to read rgo json: {exc}')
            return

        stamp = self.get_clock().now().to_msg()
        if not data.get('has_state', False):
            self._delete_rgo_markers(stamp)
            self.last_rgo_mtime_ns = mtime_ns
            return

        body_center = data.get('body_center_m')
        if body_center is None and 'state' in data and len(data['state']) >= 5:
            body_center = [data['state'][0], data['state'][2], data['state'][4]]
        if body_center is None:
            self.get_logger().warning('rgo json missing body_center_m/state')
            self.last_rgo_mtime_ns = mtime_ns
            return

        center_pos = cv_camera_point_to_visual(body_center)
        center = Marker()
        center.header.stamp = stamp
        center.header.frame_id = 'camera_link'
        center.ns = 'rgo_center'
        center.id = 1
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position.x = center_pos[0]
        center.pose.position.y = center_pos[1]
        center.pose.position.z = center_pos[2]
        center.pose.orientation.w = 1.0
        center.scale.x = 0.08
        center.scale.y = 0.08
        center.scale.z = 0.08
        center.color.r = 1.0
        center.color.g = 0.35
        center.color.b = 0.1
        center.color.a = 0.95
        self.rgo_center_pub.publish(center)

        slot_colors = {
            'FRONT': (0.1, 1.0, 0.2),
            'LEFT': (0.2, 0.5, 1.0),
            'BACK': (1.0, 0.85, 0.1),
            'RIGHT': (1.0, 0.1, 0.8),
        }
        for idx, armor_data in enumerate(data.get('armors', [])):
            pos = cv_camera_point_to_visual(armor_data['center_m'])
            quat = cv_yaw_pitch_to_visual_quat(
                armor_data.get('yaw_rad', 0.0),
                armor_data.get('pitch_rad', 0.0),
            )
            slot = armor_data.get('slot', 'UNKNOWN')
            color = slot_colors.get(slot, (0.8, 0.8, 0.8))

            armor = Marker()
            armor.header.stamp = stamp
            armor.header.frame_id = 'camera_link'
            armor.ns = 'rgo_armor'
            armor.id = idx
            armor.type = Marker.CUBE
            armor.action = Marker.ADD
            armor.pose.position.x = pos[0]
            armor.pose.position.y = pos[1]
            armor.pose.position.z = pos[2]
            armor.pose.orientation.x = quat[0]
            armor.pose.orientation.y = quat[1]
            armor.pose.orientation.z = quat[2]
            armor.pose.orientation.w = quat[3]
            armor.scale.x = 0.02
            armor.scale.y = 0.14
            armor.scale.z = 0.06
            armor.color.r = color[0]
            armor.color.g = color[1]
            armor.color.b = color[2]
            armor.color.a = 0.75
            self.rgo_armor_pub.publish(armor)

        self.last_rgo_mtime_ns = mtime_ns


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
