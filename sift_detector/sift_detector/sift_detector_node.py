#!/usr/bin/env python3
"""
SIFT Object Detector Node for ROS2
Uses SIFT template matching with RealSense depth camera to detect objects
and publish their 3D positions in world coordinates.
"""

import sys
sys.path.insert(0, '/home/haowen/yolov5_env/lib/python3.10/site-packages')

import cv2
import numpy as np
from pathlib import Path
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
import pyrealsense2 as rs
import json


class SIFTObjectDetectorNode(Node):
    """ROS2 Node for SIFT-based object detection with 3D position publishing"""
    
    def __init__(self):
        super().__init__('sift_object_detector_node')
        
        # 声明ROS2参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('template_dir', '/home/haowen/BUAARobertTeam/abu robocon 2026 map/贴图/R2 KFS图案/'),
                ('num_templates', 15),
                ('angle_thresh', 15.0),
                ('aspect_thresh', 0.3),
                ('overlap_thresh', 50.0),
                ('min_good_matches', 8),
                ('lowe_ratio', 0.8),
                ('fx', 608.692),
                ('fy', 608.622),
                ('cx', 328.773),
                ('cy', 243.008),
                ('camera_x', 0.0),
                ('camera_y', 0.0),
                ('camera_z', 0.0),
            ]
        )
        
        # 获取参数
        self.template_dir = self.get_parameter('template_dir').value
        self.num_templates = self.get_parameter('num_templates').value
        self.angle_thresh = self.get_parameter('angle_thresh').value
        self.aspect_thresh = self.get_parameter('aspect_thresh').value
        self.overlap_thresh = self.get_parameter('overlap_thresh').value
        self.min_good_matches = self.get_parameter('min_good_matches').value
        self.lowe_ratio = self.get_parameter('lowe_ratio').value
        
        # 相机内参矩阵
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 相机外参矩阵（相机在世界坐标系中的位置）
        camera_x = self.get_parameter('camera_x').value
        camera_y = self.get_parameter('camera_y').value
        camera_z = self.get_parameter('camera_z').value
        
        self.extrinsic_matrix = np.array([
            [1, 0, 0, camera_x],
            [0, 1, 0, camera_y],
            [0, 0, 1, camera_z],
            [0, 0, 0, 1]
        ])
        
        self.extrinsic_inv = np.array([
            [1, 0, 0, -camera_x],
            [0, 1, 0, -camera_y],
            [0, 0, 1, -camera_z],
            [0, 0, 0, 1]
        ])
        
        self.get_logger().info(f"Camera intrinsic matrix:\n{self.K}")
        self.get_logger().info(f"Camera extrinsic matrix:\n{self.extrinsic_matrix}")
        
        # 初始化SIFT检测器
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        
        # 加载模板
        self.templates = []
        self.kp_des_list = []
        self.load_templates()
        
        # 初始化RealSense相机
        self.setup_realsense()
        
        # 创建ROS2发布者
        self.position_publisher = self.create_publisher(
            PointStamped, 
            'object_position', 
            10
        )
        self.detection_info_publisher = self.create_publisher(
            String,
            'detection_info',
            10
        )
        
        # 创建定时器（30fps）
        self.timer = self.create_timer(0.033, self.process_frame)
        
        self.get_logger().info("SIFT Object Detector Node initialized successfully!")
        self.get_logger().info("=" * 80)
    
    def load_templates(self):
        """加载所有模板图像并提取SIFT特征"""
        self.get_logger().info(f"Loading {self.num_templates} templates from {self.template_dir}")
        
        for i in range(1, self.num_templates + 1):
            template_path = f"{self.template_dir}{i}.png"
            tpl = cv2.imread(template_path, 0)
            
            if tpl is None:
                self.get_logger().warn(f"Failed to load template: {template_path}")
                self.templates.append(None)
                self.kp_des_list.append(([], None))
                continue
            
            # 提取SIFT特征
            kp, des = self.sift.detectAndCompute(tpl, None)
            self.templates.append(tpl)
            self.kp_des_list.append((kp, des))
            
            self.get_logger().info(f"✓ Loaded template {i}: {tpl.shape}, {len(kp)} keypoints")
        
        self.get_logger().info(f"Successfully loaded {len([t for t in self.templates if t is not None])} templates")
    
    def setup_realsense(self):
        """初始化RealSense相机"""
        self.get_logger().info("Initializing RealSense camera...")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置彩色流和深度流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 启动相机
        self.pipeline.start(config)
        
        # 获取深度缩放系数
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 创建对齐对象（深度对齐到彩色）
        self.align = rs.align(rs.stream.color)
        
        self.get_logger().info(f"RealSense camera initialized (depth scale: {self.depth_scale})")
    
    def is_rectangle(self, pts):
        """判断四个点是否构成矩形"""
        pts = pts.reshape(4, 2)
        
        # 计算四条边长度
        edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        
        # 计算对角线长度
        diag1 = np.linalg.norm(pts[0] - pts[2])
        diag2 = np.linalg.norm(pts[1] - pts[3])
        
        # 计算四个角度
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        angles = [angle(pts[i-1], pts[i], pts[(i+1)%4]) for i in range(4)]
        
        # 判断角度接近90°，对角线长度接近
        if all(abs(a - 90) < self.angle_thresh for a in angles) and \
           abs(diag1 - diag2) / max(diag1, diag2) < self.aspect_thresh:
            return True
        return False
    
    def rect_center(self, rect):
        """计算矩形中心点"""
        pts = rect.reshape(4, 2)
        return np.mean(pts, axis=0)
    
    def is_overlap(self, rect1, rect2):
        """判断两个矩形是否重叠"""
        c1 = self.rect_center(rect1)
        c2 = self.rect_center(rect2)
        return np.linalg.norm(c1 - c2) < self.overlap_thresh
    
    def pixel_to_world(self, u, v, depth):
        """
        将像素坐标转换为世界坐标系下的3D坐标
        
        Args:
            u: 像素x坐标
            v: 像素y坐标
            depth: 深度值（mm）
        
        Returns:
            世界坐标系下的3D坐标 [x, y, z] (mm)
        """
        # 像素坐标 -> 相机坐标系
        x_camera = (u - self.K[0, 2]) * depth / self.K[0, 0]
        y_camera = (v - self.K[1, 2]) * depth / self.K[1, 1]
        z_camera = depth
        
        point_camera = np.array([x_camera, y_camera, z_camera, 1.0])
        
        # 相机坐标系 -> 世界坐标系
        point_world = self.extrinsic_inv @ point_camera
        
        return point_world[:3]
    
    def detect_objects(self, gray_frame, kp_frame, des_frame):
        """
        使用SIFT模板匹配检测物体
        
        Returns:
            检测结果列表，每个结果包含：rect, score, idx, center
        """
        results = []
        
        for idx, (kp_tpl, des_tpl) in enumerate(self.kp_des_list):
            template = self.templates[idx]
            
            if des_frame is None or des_tpl is None or template is None:
                continue
            
            # 特征匹配
            matches = self.bf.knnMatch(des_tpl, des_frame, k=2)
            
            # Lowe's ratio test
            good = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.lowe_ratio * n.distance:
                        good.append(m)
            
            # 如果匹配点足够多
            if len(good) > self.min_good_matches:
                src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                # 计算单应性矩阵
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = template.shape
                    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # 判断是否为矩形
                    if self.is_rectangle(dst):
                        center = self.rect_center(dst)
                        results.append({
                            'rect': dst,
                            'score': len(good),
                            'idx': idx,
                            'center': center
                        })
        
        return results
    
    def filter_overlapping_detections(self, results):
        """过滤重叠的检测结果，保留得分最高的"""
        # 按得分降序排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results = []
        for res in results:
            overlap = False
            for kept in final_results:
                if self.is_overlap(res['rect'], kept['rect']):
                    overlap = True
                    break
            if not overlap:
                final_results.append(res)
        
        return final_results
    
    def process_frame(self):
        """处理每一帧图像"""
        try:
            # 获取RealSense帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.get_logger().warn("No frame received from RealSense")
                return
            
            # 转换为numpy数组
            frame_rgb = np.asanyarray(color_frame.get_data())
            frame_depth = np.asanyarray(depth_frame.get_data())
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            
            # 提取SIFT特征
            kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
            
            # 检测物体
            results = self.detect_objects(gray, kp_frame, des_frame)
            
            # 过滤重叠检测
            final_results = self.filter_overlapping_detections(results)
            
            # 可视化
            vis_frame = frame_rgb.copy()
            
            if len(final_results) == 0:
                # 没有检测到物体
                self.get_logger().info("No objects detected in current frame")
                self.publish_position(None, None)
            else:
                # 处理每个检测结果
                self.get_logger().info("=" * 80)
                self.get_logger().info(f"✓ Detected {len(final_results)} object(s) in current frame")
                
                for i, res in enumerate(final_results):
                    template_id = res['idx'] + 1
                    score = res['score']
                    center = res['center']
                    rect = res['rect']
                    
                    # 获取中心点像素坐标
                    center_x = int(center[0])
                    center_y = int(center[1])
                    
                    # 获取深度值
                    if 0 <= center_x < frame_depth.shape[1] and 0 <= center_y < frame_depth.shape[0]:
                        depth_raw = frame_depth[center_y, center_x]
                        depth_value = float(depth_raw) * self.depth_scale * 1000  # 转换为mm
                        
                        if depth_value > 0:
                            # 转换为世界坐标
                            world_coords = self.pixel_to_world(center_x, center_y, depth_value)
                            
                            # 详细日志输出
                            #修改日志输出
                           # self.get_logger().info(f"")
                            self.get_logger().info(f"  Object #{i+1}:")
                            self.get_logger().info(f"    ├─ Template ID: {template_id}")
                            self.get_logger().info(f"    ├─ Match Score: {score} good matches")
                            #self.get_logger().info(f"    ├─ Pixel Position: ({center_x}, {center_y})")
                            #self.get_logger().info(f"    ├─ Depth: {depth_value:.2f} mm")
                            self.get_logger().info(f"    └─ World Position: X={world_coords[0]:.2f} mm, Y={world_coords[1]:.2f} mm, Z={world_coords[2]:.2f} mm")
                            
                            # 发布位置信息
                            self.publish_position(world_coords, template_id, score)
                            
                            # 可视化：绘制矩形框
                            vis_frame = cv2.polylines(
                                vis_frame, 
                                [np.int32(rect)], 
                                True, 
                                (0, 255, 0), 
                                3, 
                                cv2.LINE_AA
                            )
                            
                            # 绘制中心点
                            cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            # 绘制文本信息
                            cv2.putText(
                                vis_frame,
                                f'Template {template_id} (Score: {score})',
                                tuple(np.int32(rect[0][0])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
                            
                            cv2.putText(
                                vis_frame,
                                f'World: ({world_coords[0]:.0f}, {world_coords[1]:.0f}, {world_coords[2]:.0f}) mm',
                                (center_x - 100, center_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                2
                            )
                        else:
                            self.get_logger().warn(f"  Invalid depth at center of Template {template_id}")
                    else:
                        self.get_logger().warn(f"  Center out of bounds for Template {template_id}")
                
                self.get_logger().info("=" * 80)
            
            # 显示结果
            cv2.imshow("SIFT Object Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("User requested shutdown (pressed 'q')")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {str(e)}")
    
    def publish_position(self, position, template_id=None, score=None):
        """
        发布物体的3D位置
        
        Args:
            position: 世界坐标系下的3D坐标 [x, y, z] (mm)
            template_id: 模板ID
            score: 匹配得分
        """
        # 发布PointStamped消息
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        if position is not None:
            msg.point.x = float(position[0])
            msg.point.y = float(position[1])
            msg.point.z = float(position[2])
        else:
            msg.point.x = float('nan')
            msg.point.y = float('nan')
            msg.point.z = float('nan')
        
        self.position_publisher.publish(msg)
        
        # 发布检测信息（JSON格式）
        if position is not None and template_id is not None:
            info = {
                'template_id': template_id,
                'score': score,
                'position': {
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                },
                'timestamp': self.get_clock().now().to_msg().sec
            }
            info_msg = String()
            info_msg.data = json.dumps(info)
            self.detection_info_publisher.publish(info_msg)
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info("Shutting down SIFT Object Detector Node...")
        try:
            self.pipeline.stop()
            cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = SIFTObjectDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()