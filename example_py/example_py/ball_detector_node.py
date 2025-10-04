#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/haowen/yolov5_env/lib/python3.10/site-packages')
import cv2
import numpy as np
from pathlib import Path
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import pyrealsense2 as rs

# yolov8模型导入
from ultralytics import YOLO

class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector_node')
        self.publisher_ = self.create_publisher(PointStamped, 'ball_position', 10)
        self.UNIT = 'mm'
        self.get_logger().info(f"Using unit: {self.UNIT}")

        # yolov8模型权重路径
        weights = "/home/haowen/BUAARobertTeam/第二周/python学习/yolo11n.pt"
        self.model = YOLO(weights)
        self.get_logger().info("Loaded YOLOv8 model")

        # 直接使用您提供的相机内参
        self.K = np.array([
            [608.692, 0, 328.773],
            [0, 608.622, 243.008],
            [0, 0, 1]
        ])
        self.get_logger().info(f"Using camera intrinsic matrix:\n{self.K}")
        
        
        # 设置外参矩阵（相机在世界坐标系中的位置和姿态）
        # 相机位置: (100, 100, 100)mm，无旋转
        self.extrinsic_matrix = np.array([
            [1, 0, 0, 100],
            [0, 1, 0, 100],
            [0, 0, 1, 100],
            [0, 0, 0, 1]
        ])
        
        # 计算外参矩阵的逆
        self.extrinsic_inv = np.array([
            [1, 0, 0, -100],
            [0, 1, 0, -100],
            [0, 0, 1, -100],
            [0, 0, 0, 1]
        ])
        
        self.get_logger().info(f"Using extrinsic matrix:\n{self.extrinsic_matrix}")
        self.get_logger().info(f"Using extrinsic inverse matrix:\n{self.extrinsic_inv}")

        """
        # 假设外参矩阵为单位矩阵（相机坐标系与世界坐标系重合）
        self.extrinsic_inv = np.eye(4)
        self.get_logger().info("Using identity extrinsic matrix")
        """

        # 初始化RealSense相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # 获取深度缩放系数
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f"Depth scale: {self.depth_scale}")

        # 创建对齐对象（深度对齐到彩色）
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.timer = self.create_timer(0.033, self.process_frame)  # ~30fps

    def pixel_to_world(self, u, v, depth):
        # 将像素坐标转换为相机坐标系下的3D坐标
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        point_camera = np.array([x * depth, y * depth, depth, 1.0])
        
        # 转换到世界坐标系
        point_world = self.extrinsic_inv @ point_camera
        return point_world[:3]

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        
        # 对齐深度帧到彩色帧
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            self.get_logger().warning("No frame received from RealSense")
            return

        frame_rgb = np.asanyarray(color_frame.get_data())
        frame_depth = np.asanyarray(depth_frame.get_data())  # uint16

        vis_frame = frame_rgb.copy()
        detected = False

        # YOLOv8推理
        results = self.model(frame_rgb, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # 检测球（COCO类别37）
            if cls == 39:  # 37对应'sports ball'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # 获取深度值（单位：米），再转为毫米
                if 0 <= center_x < frame_depth.shape[1] and 0 <= center_y < frame_depth.shape[0]:
                    depth_raw = frame_depth[center_y, center_x]
                    depth_value = float(depth_raw) * self.depth_scale * 1000  # mm
                    
                    # 跳过无效深度值
                    if depth_value <= 0:
                        continue
                        
                    world_coords = self.pixel_to_world(
                        center_x,
                        center_y,
                        depth_value
                    )
                    # 可视化：画框、中心点、类别和坐标
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(vis_frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"({world_coords[0]:.1f},{world_coords[1]:.1f},{world_coords[2]:.1f})mm",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    self.publish_position(world_coords)
                    detected = True
                    break
        if not detected:
            self.publish_position(None)

        # 可视化窗口
        cv2.imshow("Ball Detection", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("Ball Detection")

    def publish_position(self, position):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"  # 改为相机坐标系
        if position is not None:
            msg.point.x = float(position[0])
            msg.point.y = float(position[1])
            msg.point.z = float(position[2])
            self.get_logger().info(f"Detected ball at: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) mm")
        else:
            msg.point.x = float('nan')
            msg.point.y = float('nan')
            msg.point.z = float('nan')
            self.get_logger().info("No ball detected")
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()