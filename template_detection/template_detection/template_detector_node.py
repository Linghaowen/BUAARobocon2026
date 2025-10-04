#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/haowen/yolov5_env/lib/python3.10/site-packages')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
import pyrealsense2 as rs


class TemplateDetectorDepthNode(Node):
    def __init__(self):
        super().__init__('template_detector_depth')
        
        # ==================== 参数声明 ====================
        # 相机参数
        self.declare_parameter('rgb_camera_index', 4)
        self.declare_parameter('depth_camera_index', 2)
        self.declare_parameter('use_realsense', True)  # 是否使用RealSense相机
        
        # 模板检测参数
        self.declare_parameter('template_base_path', '/home/haowen/BUAARobertTeam/abu robocon 2026 map/贴图/R2 KFS图案')
        self.declare_parameter('lower_blue', [90, 50, 50])
        self.declare_parameter('upper_blue', [130, 255, 255])
        self.declare_parameter('min_contour_area', 1000)
        self.declare_parameter('angle_thresh', 15)
        self.declare_parameter('aspect_thresh', 0.3)
        self.declare_parameter('overlap_thresh', 50)
        
        # 相机内参
        self.declare_parameter('fx', 608.692)
        self.declare_parameter('fy', 608.622)
        self.declare_parameter('cx', 328.773)
        self.declare_parameter('cy', 243.008)
        
        # 相机外参 (4x4矩阵，行优先)
        self.declare_parameter('extrinsic_matrix', [
            1.0, 0.0, 0.0, 100.0,
            0.0, 1.0, 0.0, 100.0,
            0.0, 0.0, 1.0, 100.0,
            0.0, 0.0, 0.0, 1.0
        ])
        
        # 可视化参数
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('show_depth_map', False)
        self.declare_parameter('show_fps', True)
        
        # ==================== 获取参数 ====================
        rgb_camera_index = self.get_parameter('rgb_camera_index').value
        depth_camera_index = self.get_parameter('depth_camera_index').value
        self.use_realsense = self.get_parameter('use_realsense').value
        template_base_path = self.get_parameter('template_base_path').value
        lower_blue = self.get_parameter('lower_blue').value
        upper_blue = self.get_parameter('upper_blue').value
        
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.angle_thresh = self.get_parameter('angle_thresh').value
        self.aspect_thresh = self.get_parameter('aspect_thresh').value
        self.overlap_thresh = self.get_parameter('overlap_thresh').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.show_depth_map = self.get_parameter('show_depth_map').value
        self.show_fps = self.get_parameter('show_fps').value
        
        self.lower_blue = np.array(lower_blue)
        self.upper_blue = np.array(upper_blue)
        
        # ==================== 相机内参矩阵 ====================
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.get_logger().info(f"Camera intrinsic matrix:\n{self.K}")
        
        # ==================== 相机外参矩阵 ====================
        extrinsic_list = self.get_parameter('extrinsic_matrix').value
        self.extrinsic_matrix = np.array(extrinsic_list).reshape(4, 4)
        
        # 计算外参逆矩阵（用于从相机坐标系转换到世界坐标系）
        self.extrinsic_inv = np.linalg.inv(self.extrinsic_matrix)
        
        self.get_logger().info(f"Extrinsic matrix:\n{self.extrinsic_matrix}")
        self.get_logger().info(f"Extrinsic inverse:\n{self.extrinsic_inv}")
        
        # ==================== ROS2 发布者 ====================
        self.detection_2d_pub = self.create_publisher(String, 'template_detections_2d', 10)
        self.detection_3d_pub = self.create_publisher(String, 'template_detections_3d', 10)
        self.point_pub = self.create_publisher(PointStamped, 'template_position', 10)
        self.image_pub = self.create_publisher(Image, 'detection_image', 10)
        
        # ==================== CV Bridge ====================
        self.bridge = CvBridge()
        
        # ==================== 统计信息 ====================
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.total_detections = 0
        
        # ==================== 加载SIFT模板 ====================
        self.get_logger().info('Loading templates...')
        template_paths = [f'{template_base_path}/{i}.png' for i in range(1, 16)]
        self.templates = [cv2.imread(path, 0) for path in template_paths]
        
        loaded_count = sum(1 for t in self.templates if t is not None)
        self.get_logger().info(f'Loaded {loaded_count}/15 templates')
        
        # 初始化SIFT
        self.sift = cv2.SIFT_create()
        self.kp_des_list = [self.sift.detectAndCompute(tpl, None) if tpl is not None else ([], None) 
                            for tpl in self.templates]
        self.bf = cv2.BFMatcher()
        
        # ==================== 初始化相机 ====================
        if self.use_realsense:
            self.init_realsense()
        else:
            self.init_separate_cameras(rgb_camera_index, depth_camera_index)
        
        # ==================== 可视化窗口 ====================
        if self.enable_visualization:
            cv2.namedWindow('Template Detection', cv2.WINDOW_NORMAL)
            if self.show_depth_map:
                cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        
        # ==================== 定时器 ====================
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 FPS
        
    def init_realsense(self):
        """初始化RealSense相机（RGB和深度在同一设备）"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)
            
            # 获取深度缩放系数
            profile = self.pipeline.get_active_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.get_logger().info(f"RealSense depth scale: {self.depth_scale}")
            
            # 创建对齐对象（深度对齐到彩色）
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            self.get_logger().info('RealSense camera initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize RealSense: {e}')
            raise
    
    def init_separate_cameras(self, rgb_index, depth_index):
        """初始化分离的RGB和深度相机"""
        self.cap_rgb = cv2.VideoCapture(rgb_index)
        self.cap_depth = cv2.VideoCapture(depth_index)
        
        if not self.cap_rgb.isOpened():
            self.get_logger().error(f'Cannot open RGB camera {rgb_index}')
            raise RuntimeError(f'RGB camera {rgb_index} failed')
        
        if not self.cap_depth.isOpened():
            self.get_logger().error(f'Cannot open depth camera {depth_index}')
            raise RuntimeError(f'Depth camera {depth_index} failed')
        
        # 设置分辨率
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.depth_scale = 0.001  # 默认深度缩放（假设深度相机输出单位为mm）
        
        self.get_logger().info(f'RGB camera {rgb_index} and depth camera {depth_index} opened')
    
    def get_aligned_frames(self):
        """获取对齐的RGB和深度帧"""
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            frame_rgb = np.asanyarray(color_frame.get_data())
            frame_depth = np.asanyarray(depth_frame.get_data())
            return frame_rgb, frame_depth
        else:
            ret_rgb, frame_rgb = self.cap_rgb.read()
            ret_depth, frame_depth = self.cap_depth.read()
            
            if not ret_rgb or not ret_depth:
                return None, None
            
            # 如果深度图是彩色的，转换为灰度
            if len(frame_depth.shape) == 3:
                frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
            
            return frame_rgb, frame_depth
    
    def pixel_to_world(self, u, v, depth_mm):
        """将像素坐标和深度值转换为世界坐标系3D坐标"""
        # 将像素坐标转换为相机坐标系下的3D坐标
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        point_camera = np.array([x * depth_mm, y * depth_mm, depth_mm, 1.0])
        
        # 转换到世界坐标系
        point_world = self.extrinsic_inv @ point_camera
        return point_world[:3]
    
    def get_depth_value(self, frame_depth, x, y):
        """获取指定像素的深度值（单位：mm）"""
        if 0 <= x < frame_depth.shape[1] and 0 <= y < frame_depth.shape[0]:
            if self.use_realsense:
                depth_raw = frame_depth[y, x]
                depth_mm = float(depth_raw) * self.depth_scale * 1000  # 转换为mm
            else:
                # 对于普通深度相机，假设输出已经是深度值
                depth_mm = float(frame_depth[y, x])
            
            return depth_mm if depth_mm > 0 else None
        return None
    
    # ==================== SIFT检测相关方法 ====================
    
    def is_rectangle(self, pts):
        """判断四个点是否构成矩形"""
        pts = pts.reshape(4, 2)
        edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        diag1 = np.linalg.norm(pts[0] - pts[2])
        diag2 = np.linalg.norm(pts[1] - pts[3])
        
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        angles = [angle(pts[i-1], pts[i], pts[(i+1)%4]) for i in range(4)]
        
        if all(abs(a-90) < self.angle_thresh for a in angles) and \
           abs(diag1-diag2)/max(diag1, diag2) < self.aspect_thresh:
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
    
    def draw_info_panel(self, frame, detections_2d, detections_3d):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        panel_height = 180
        panel_width = 400
        
        # 创建半透明面板
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 绘制文本信息
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # FPS
        if self.show_fps:
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (20, y_offset), 
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += 30
        
        # 检测到的模板数量
        cv2.putText(frame, f'Templates Detected: {len(detections_2d)}', (20, y_offset), 
                   font, font_scale, (0, 255, 255), thickness)
        y_offset += 30
        
        # 检测到的模板ID列表
        if detections_2d:
            template_ids = ', '.join([f'T{d["template_id"]}' for d in detections_2d])
            cv2.putText(frame, f'IDs: {template_ids}', (20, y_offset), 
                       font, 0.5, color, 1)
            y_offset += 25
        
        # 3D坐标信息
        if detections_3d:
            cv2.putText(frame, f'3D Coordinates Available', (20, y_offset), 
                       font, 0.5, (0, 255, 0), 1)
            y_offset += 25
        
        # 总检测次数
        cv2.putText(frame, f'Total Detections: {self.total_detections}', (20, y_offset), 
                   font, 0.5, color, 1)
        
        # 控制提示
        help_text = "Press 'q' to quit | 's' to save"
        text_size = cv2.getTextSize(help_text, font, 0.5, 1)[0]
        cv2.putText(frame, help_text, (w - text_size[0] - 10, 25), 
                   font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_detection(self, frame, res, center_x, center_y, world_coords=None):
        """绘制单个检测结果"""
        if res['type'] == 'circle':
            color = (0, 255, 255)  # 黄色
            center_tuple = tuple(map(int, res['center']))
            radius = int(res['radius'])
            cv2.circle(frame, center_tuple, radius, color, 3)
            cv2.circle(frame, center_tuple, 5, (0, 0, 255), -1)
            
            label = f"T{res['idx']+1}"
            text_pos = (center_tuple[0] - radius, center_tuple[1] - radius - 50)
        else:
            color = (0, 255, 0)  # 绿色
            frame = cv2.polylines(frame, [res['rect'].astype(np.int32)], True, color, 3, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 绘制矩形四角
            pts = res['rect'].astype(np.int32).reshape(4, 2)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 3, (255, 0, 255), -1)
            
            label = f"T{res['idx']+1}"
            text_pos = tuple(res['rect'][0][0].astype(np.int32))
        
        # 绘制标签背景和文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (text_pos[0] - 5, text_pos[1] - text_h - 5),
                     (text_pos[0] + text_w + 5, text_pos[1] + 5),
                     (0, 0, 0), -1)
        cv2.putText(frame, label, text_pos, font, font_scale, color, thickness)
        
        # 绘制2D坐标
        info_2d = f"2D: ({center_x},{center_y})"
        info_pos = (text_pos[0], text_pos[1] + 25)
        (info_w, info_h), _ = cv2.getTextSize(info_2d, font, 0.5, 1)
        cv2.rectangle(frame, 
                     (info_pos[0] - 3, info_pos[1] - info_h - 3),
                     (info_pos[0] + info_w + 3, info_pos[1] + 3),
                     (0, 0, 0), -1)
        cv2.putText(frame, info_2d, info_pos, font, 0.5, (255, 255, 255), 1)
        
        # 绘制3D坐标
        if world_coords is not None:
            info_3d = f"3D: ({world_coords[0]:.0f},{world_coords[1]:.0f},{world_coords[2]:.0f})mm"
            info_pos_3d = (text_pos[0], text_pos[1] + 45)
            (info_w, info_h), _ = cv2.getTextSize(info_3d, font, 0.5, 1)
            cv2.rectangle(frame, 
                         (info_pos_3d[0] - 3, info_pos_3d[1] - info_h - 3),
                         (info_pos_3d[0] + info_w + 3, info_pos_3d[1] + 3),
                         (0, 0, 0), -1)
            cv2.putText(frame, info_3d, info_pos_3d, font, 0.5, (0, 255, 255), 1)
        
        return frame
    
    # ==================== 主处理循环 ====================
    
    def timer_callback(self):
        """主处理回调函数"""
        # 更新FPS
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = time.time()
            self.fps = self.frame_count / (end_time - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()
        
        # 获取对齐的RGB和深度帧
        frame_rgb, frame_depth = self.get_aligned_frames()
        
        if frame_rgb is None or frame_depth is None:
            self.get_logger().warn('Failed to get frames')
            return
        
        # ========== 蓝色区域检测 ==========
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_contour_area:
                continue
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and self.is_rectangle(approx):
                blue_rects.append(box)
        
        # ========== SIFT模板匹配 ==========
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        out_frame = frame_rgb.copy()
        results = []
        
        if not blue_rects:
            if self.enable_visualization:
                out_frame = self.draw_info_panel(out_frame, [], [])
                cv2.imshow('Template Detection', out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rclpy.shutdown()
            
            img_msg = self.bridge.cv2_to_imgmsg(out_frame, encoding='bgr8')
            self.image_pub.publish(img_msg)
            return
        
        # 在每个蓝色区域内进行SIFT匹配
        for rect in blue_rects:
            mask_rect = np.zeros_like(gray)
            cv2.fillPoly(mask_rect, [rect], 255)
            
            x, y, w, h = cv2.boundingRect(rect)
            roi = gray[y:y+h, x:x+w]
            
            kp_rect = []
            des_rect = None
            if kp_frame is not None and des_frame is not None:
                for i, kp in enumerate(kp_frame):
                    pt = tuple(map(int, kp.pt))
                    if pt[1] < mask_rect.shape[0] and pt[0] < mask_rect.shape[1]:
                        if mask_rect[pt[1], pt[0]] == 255:
                            kp_rect.append(kp)
                            
                if kp_rect:
                    indices = [i for i, kp in enumerate(kp_frame) if kp in kp_rect]
                    des_rect = des_frame[indices]
            
            if des_rect is None or len(kp_rect) == 0:
                continue
            
            best_match = None
            best_score = 0
            
            # SIFT模板匹配
            for idx, (kp_tpl, des_tpl) in enumerate(self.kp_des_list):
                if idx == 13:  # 跳过模板14
                    continue
                
                template = self.templates[idx]
                if des_rect is None or des_tpl is None or template is None:
                    continue
                
                matches = self.bf.knnMatch(des_tpl, des_rect, k=2)
                good = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.8 * n.distance:
                            good.append(m)
                
                if len(good) > 8 and len(good) > best_score:
                    src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_rect[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h_tpl, w_tpl = template.shape
                        pts = np.float32([[0, 0], [w_tpl, 0], [w_tpl, h_tpl], [0, h_tpl]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        if self.is_rectangle(dst):
                            best_match = {
                                'rect': dst,
                                'score': len(good),
                                'idx': idx,
                                'type': 'sift'
                            }
                            best_score = len(good)
            
            if best_match is not None:
                results.append(best_match)
            else:
                # 圆形检测（模板14）
                idx = 13
                template = self.templates[idx]
                if template is not None:
                    roi_blur = cv2.GaussianBlur(roi, (9, 9), 2)
                    circles = cv2.HoughCircles(
                        roi_blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist=50,
                        param1=100,
                        param2=30,
                        minRadius=10,
                        maxRadius=100
                    )
                    
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        circle = circles[0][0]
                        cx = x + circle[0]
                        cy = y + circle[1]
                        radius = circle[2]
                        
                        rect_circle = np.array([
                            [cx - radius, cy - radius],
                            [cx + radius, cy - radius],
                            [cx + radius, cy + radius],
                            [cx - radius, cy + radius]
                        ])
                        
                        results.append({
                            'rect': rect_circle.reshape(4, 1, 2).astype(np.float32),
                            'score': 100,
                            'idx': idx,
                            'type': 'circle',
                            'center': (cx, cy),
                            'radius': radius
                        })
        
        # 去重
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
        
        # ========== 生成检测结果（2D和3D） ==========
        detections_2d = []
        detections_3d = []
        
        for res in final_results:
            center = self.rect_center(res['rect'])
            center_x, center_y = int(center[0]), int(center[1])
            
            # 2D检测数据
            detection_2d = {
                'template_id': res['idx'] + 1,
                'type': res['type'],
                'center_x': center_x,
                'center_y': center_y,
                'score': res['score']
            }
            detections_2d.append(detection_2d)
            
            # 获取深度值并计算3D坐标
            depth_mm = self.get_depth_value(frame_depth, center_x, center_y)
            
            if depth_mm is not None and depth_mm > 0:
                world_coords = self.pixel_to_world(center_x, center_y, depth_mm)
                
                detection_3d = {
                    'template_id': res['idx'] + 1,
                    'type': res['type'],
                    'world_x': float(world_coords[0]),
                    'world_y': float(world_coords[1]),
                    'world_z': float(world_coords[2]),
                    'depth_mm': float(depth_mm),
                    'score': res['score']
                }
                detections_3d.append(detection_3d)
                
                # 绘制检测结果（包含3D坐标）
                out_frame = self.draw_detection(out_frame, res, center_x, center_y, world_coords)
                
                # 发布单个点的3D坐标
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "world"
                point_msg.point.x = float(world_coords[0])
                point_msg.point.y = float(world_coords[1])
                point_msg.point.z = float(world_coords[2])
                self.point_pub.publish(point_msg)
            else:
                # 只绘制2D检测结果
                out_frame = self.draw_detection(out_frame, res, center_x, center_y)
        
        # 更新统计
        if detections_2d:
            self.total_detections += len(detections_2d)
        
        # ========== 发布检测结果 ==========
        if detections_2d:
            # 发布2D检测结果
            msg_2d = String()
            msg_2d.data = json.dumps(detections_2d)
            self.detection_2d_pub.publish(msg_2d)
            
            # 终端输出2D信息
            self.get_logger().info(
                f'Detected {len(detections_2d)} templates (2D): ' +
                ', '.join([f"T{d['template_id']}({d['center_x']},{d['center_y']})" 
                          for d in detections_2d])
            )
        
        if detections_3d:
            # 发布3D检测结果
            msg_3d = String()
            msg_3d.data = json.dumps(detections_3d)
            self.detection_3d_pub.publish(msg_3d)
            
            # 终端输出3D信息
            for d in detections_3d:
                self.get_logger().info(
                    f"Template {d['template_id']} (3D): "
                    f"World=({d['world_x']:.1f}, {d['world_y']:.1f}, {d['world_z']:.1f})mm, "
                    f"Depth={d['depth_mm']:.1f}mm, Score={d['score']}"
                )
        
        # ========== 可视化 ==========
        if self.enable_visualization:
            out_frame = self.draw_info_panel(out_frame, detections_2d, detections_3d)
            cv2.imshow('Template Detection', out_frame)
            
            if self.show_depth_map:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(frame_depth, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth Map', depth_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('s'):
                filename = f'detection_{int(time.time())}.jpg'
                cv2.imwrite(filename, out_frame)
                self.get_logger().info(f'Saved frame to {filename}')
        
        # 发布可视化图像
        img_msg = self.bridge.cv2_to_imgmsg(out_frame, encoding='bgr8')
        self.image_pub.publish(img_msg)
    
    def destroy_node(self):
        """清理资源"""
        if self.use_realsense:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
        else:
            if hasattr(self, 'cap_rgb'):
                self.cap_rgb.release()
            if hasattr(self, 'cap_depth'):
                self.cap_depth.release()
        
        if self.enable_visualization:
            cv2.destroyAllWindows()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TemplateDetectorDepthNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()