from typing import Tuple, Optional, Literal, List
from copy import deepcopy
from collections import namedtuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_system_default

from ament_index_python import get_package_share_directory

from tf2_ros import Buffer, TransformBroadcaster, TransformListener, StaticTransformBroadcaster

from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from geometry_msgs.msg import TransformStamped, Pose, Transform

from ass_msgs.msg import ARMstrongKinematics, ARMstrongPlanRequest, ARMstrongPlanStatus

from armstrong_py.conversions import ros_pos_to_np_pos, ros_quat_to_rotation, list_to_ros_vec3, list_to_ros_point, rot_to_ros_quat

from bolt_fastener.bolt_detector import BoltDetector, DetectionResult
from bolt_fastener.point_cloud_processor import PointCloudProcessor


class BoltFastenerNode(Node):
    def __init__(self, name='bolt_fastener', use_tcp_tunnel=True):
        super().__init__(name)

        model = get_package_share_directory('bolt_fastener') + '/best.pt'

        self.head_detector = BoltDetector(model_path=model)
        self.head_processor = PointCloudProcessor()

        self.hand_detector = BoltDetector(model_path=model)
        self.hand_processor = PointCloudProcessor()

        tcp_tunnel_prefix = '/tcp_tunnel_client' if use_tcp_tunnel else ''
        camera_namespace = '/m4_camera'
        head_camera_name = '/head_camera'
        hand_camera_name = '/hand_camera'

        self.target = None
        self.target_pose = None
        
        self.head_depth_intrinsics = {}
        self.head_rgb_intrinsics = {}
        self.head_confidence_threshold = 0.
        self.head_detection: DetectionResult | None = None
        self.head_bolts = []
        self.head_bolts_detection_info: DetectionResult | None = None
        self.joint_state: np.ndarray | None = None

        self.head_rgb: np.ndarray | None = None
        self.head_depth: np.ndarray | None = None

        self.hand_bolts = []
        self.hand_confidence_threshold = 0.
        self.hand_detection: DetectionResult | None = None
        self.hand_bolts_detection_info: DetectionResult | None = None

        self.hand_rgb: np.ndarray | None = None
        self.hand_depth: np.ndarray | None = None
        self.hand_depth_intrinsics = {}
        self.hand_rgb_intrinsics = {}
        self.r_pose: Tuple[np.ndarray, Rotation] = (np.zeros(3), Rotation.identity())
        self.armstrong_plan_status = ARMstrongPlanStatus.IDLE
        self.docking_status: Literal['IDLE', 'DOCKING', 'DOCKED', 'FAILED'] = 'IDLE'
        self.docking_pose = None

        self.is_head_update_detection: bool = True
        self.is_hand_update_detection: bool = True
        self.is_camera_head: bool = True

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        head_cam_transform = TransformStamped()
        head_cam_transform.header.stamp = self.get_clock().now().to_msg()
        head_cam_transform.header.frame_id = 'origin'
        head_cam_transform.child_frame_id = 'head_camera_link'
        head_cam_transform.transform.translation = list_to_ros_vec3(np.array([-75.31, 0, 531.14]) / 1000)
        head_cam_transform.transform.rotation = rot_to_ros_quat(Rotation.from_rotvec([0, 30*np.pi/180, 0]))
        
        hand_cam_transform = TransformStamped()
        hand_cam_transform.header.stamp = self.get_clock().now().to_msg()
        hand_cam_transform.header.frame_id = 'r_link6'
        hand_cam_transform.child_frame_id = 'hand_camera_link'
        hand_cam_transform.transform.translation = list_to_ros_vec3(np.array([164.36, -20.5, 58.]) / 1000)
        hand_cam_transform.transform.rotation = rot_to_ros_quat(Rotation.identity())

        tf_static_broadcaster.sendTransform(head_cam_transform)
        tf_static_broadcaster.sendTransform(hand_cam_transform)

        self.rgb_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + head_camera_name + '/color/image_raw/compressed',
            callback=self.on_head_image_update,
            qos_profile=qos_profile_system_default,
            )
        self.depth_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + head_camera_name + '/aligned_depth_to_color/image_raw/compressedDepth',
            callback=self.on_depth_update,
            qos_profile=qos_profile_system_default,
            )
        self.depth_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=camera_namespace + head_camera_name + "/aligned_depth_to_color/camera_info",
            callback=self.on_head_depth_info_update,
            qos_profile=qos_profile_system_default,
            )
        self.hand_rgb_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + hand_camera_name + '/color/image_raw/compressed',
            callback=self.on_hand_image_update,
            qos_profile=qos_profile_system_default,
            )
        self.hand_depth_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + hand_camera_name + '/aligned_depth_to_color/image_raw/compressedDepth',
            callback=self.on_hand_depth_update,
            qos_profile=qos_profile_system_default,
            )
        self.hand_depth_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=camera_namespace + hand_camera_name + "/aligned_depth_to_color/camera_info",
            callback=self.on_hand_depth_info_update,
            qos_profile=qos_profile_system_default,
            )
        self.head_rgb_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=camera_namespace + head_camera_name + "/color/camera_info",
            callback=self.on_head_rgb_info_update,
            qos_profile=qos_profile_system_default,
            )
        self.hand_rgb_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=camera_namespace + hand_camera_name + "/color/camera_info",
            callback=self.on_hand_rgb_info_update,
            qos_profile=qos_profile_system_default,
            )
            
        self.joint_state_sub = self.create_subscription(
            msg_type=JointState,
            topic='joint_states',
            callback=self.on_joint_state_update,
            qos_profile=qos_profile_system_default,
        )
        self.right_arm_kinematics_sub = self.create_subscription(
            msg_type=ARMstrongKinematics,
            topic='right_arm_kinematics',
            callback=self.on_right_arm_kinematics_update,
            qos_profile=qos_profile_system_default,
        )
        self.armstrong_plan_status_sub = self.create_subscription(
            msg_type=ARMstrongPlanStatus,
            topic="armstrong_plan_status",
            callback=self.on_armstrong_plan_status_update,
            qos_profile=qos_profile_system_default,
        )
        self.armstrong_plan_pub = self.create_publisher(
            msg_type=ARMstrongPlanRequest, 
            topic='armstrong_plan_request', 
            qos_profile=qos_profile_system_default,
        )
        # cv2.namedWindow('head_depth')
        # cv2.createTrackbar('min', 'head_depth', 0, 10000, lambda x: None)
        # cv2.createTrackbar("max", 'head_depth', 0, 10000, lambda x: None)

        # cv2.setTrackbarPos('max', 'head_depth', 10000)
        
        cv2.namedWindow('head_rgb')
        cv2.createTrackbar('Contrast', 'head_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar('Brightness', 'head_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar("Confidence", 'head_rgb', 0, 100, lambda x: None)

        cv2.setTrackbarPos('Contrast', 'head_rgb', 50)
        cv2.setTrackbarPos('Brightness', 'head_rgb', 50)
        cv2.setTrackbarPos('Confidence', 'head_rgb', 50)
        cv2.setMouseCallback('head_rgb', self.on_head_mouse_event)

        cv2.namedWindow('hand_rgb')
        cv2.createTrackbar('Contrast', 'hand_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar('Brightness', 'hand_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar("Confidence", 'hand_rgb', 0, 100, lambda x: None)

        cv2.setTrackbarPos('Contrast', 'hand_rgb', 50)
        cv2.setTrackbarPos('Brightness', 'hand_rgb', 50)
        cv2.setTrackbarPos('Confidence', 'hand_rgb', 50)

        cv2.setMouseCallback('hand_rgb', self.on_hand_mouse_event)


        self.head_draw_timer = self.create_timer(1/30, self.on_head_draw_timer)
        self.head_prev_time = self.get_clock().now()
        self.head_frame_time = []
        self.head_fps = 0.

        self.hand_draw_timer = self.create_timer(1/30, self.on_hand_draw_timer)
        self.hand_prev_time = self.get_clock().now()
        self.hand_frame_time = []
        self.hand_fps = 0.
        
        self.planning_status_map = {
            ARMstrongPlanStatus.IDLE: "IDLE",
            ARMstrongPlanStatus.PLANNING: "PLANNING",
            ARMstrongPlanStatus.EXECUTING: "EXECUTING",
            ARMstrongPlanStatus.COMPLETED: "COMPLETED",
            ARMstrongPlanStatus.FAILED: "FAILED"
        }
    
    def on_head_image_update(self, msg: CompressedImage):
        contrast = cv2.getTrackbarPos('Contrast', 'head_rgb')
        brightness = cv2.getTrackbarPos("Brightness", 'head_rgb')
        self.head_confidence_threshold = cv2.getTrackbarPos("Confidence", 'head_rgb') / 100.0
        
        str_msg = msg.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                         dtype=np.uint8, buffer=msg.data)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = image.astype(np.uint8)
        
        img = cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)
        self.head_rgb = img
        
        res = self.head_detector.track(img)
        if res is not None:
            self.head_detection = res
    
    def on_hand_image_update(self, msg: CompressedImage):
        contrast = cv2.getTrackbarPos('Contrast', 'hand_rgb')
        brightness = cv2.getTrackbarPos("Brightness", 'hand_rgb')
        self.hand_confidence_threshold = cv2.getTrackbarPos("Confidence", 'hand_rgb') / 100.0
        
        str_msg = msg.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                         dtype=np.uint8, buffer=msg.data)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = image.astype(np.uint8)
        
        img = cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)
        self.hand_rgb = img

        res = self.hand_detector.track(img)
        if res is not None:
            self.hand_detection = res
        
    def on_depth_update(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        depth = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
        self.head_depth = depth.astype(np.float32) / 1000

        if (self.head_detection is not None 
            and self.head_depth_intrinsics != {} 
            and self.head_depth is not None
            and self.is_head_update_detection
            ):
            res = self.head_processor.estimate_bolt_pose_pipeline(
                depth=self.head_depth,
                detection=self.head_detection,
                conf=self.head_confidence_threshold,
                crop_by_roi=True,
                radius=0.03,
                force_single_estimation=False)
            if res is None:
                return
            self.head_bolts, self.head_bolts_detection_info = res
        for b in self.head_bolts:
            self.publish_transform(b, 'head_camera_color_optical_frame')

    def on_hand_depth_update(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        depth = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
        self.hand_depth = depth.astype(np.float32) / 1000
        self.get_logger().info(f"{self.hand_depth.shape}")
        if (self.hand_detection is not None 
            and self.hand_depth_intrinsics != {} 
            and self.hand_depth is not None
            and self.is_hand_update_detection
            ):
            res = self.hand_processor.estimate_bolt_pose_pipeline(
                depth=self.hand_depth,
                detection=self.hand_detection,
                conf=self.hand_confidence_threshold,
                crop_by_roi=True,
                radius=0.03,
                force_single_estimation=False)
            if res is None:
                return
            self.hand_bolts, self.hand_bolts_detection_info = res
        for b in self.hand_bolts:
            self.publish_transform(b, 'hand_camera_color_optical_frame')
            
    def on_head_depth_info_update(self, info: CameraInfo):
        depth_intrinsics = {'fx': info.k[0], 'fy': info.k[4], 'cx': info.k[2], 'cy': info.k[5], 'width': info.width, 'height': info.height, 'mtx': np.array(info.k).reshape(3, 3), 'dist': np.array(info.d)}
        self.head_depth_intrinsics = depth_intrinsics
        self.head_processor.set_depth_intrinsics(depth_intrinsics)
        self.head_detector.cam_info = depth_intrinsics

    def on_hand_depth_info_update(self, info: CameraInfo):
        hand_depth_intrinsics = {'fx': info.k[0], 'fy': info.k[4], 'cx': info.k[2], 'cy': info.k[5], 'width': info.width, 'height': info.height, 'mtx': np.array(info.k).reshape(3, 3), 'dist': np.array(info.d)}
        self.hand_depth_intrinsics = hand_depth_intrinsics
        self.hand_processor.set_depth_intrinsics(hand_depth_intrinsics)
        self.hand_detector.cam_info = hand_depth_intrinsics

    def on_head_rgb_info_update(self, info: CameraInfo):
        self.head_rgb_intrinsics = {'fx': info.k[0], 'fy': info.k[4], 'cx': info.k[2], 'cy': info.k[5], 'width': info.width, 'height': info.height, 'mtx': np.array(info.k).reshape(3, 3), 'dist': np.array(info.d)}

    def on_hand_rgb_info_update(self, info: CameraInfo):
        self.hand_rgb_intrinsics = {'fx': info.k[0], 'fy': info.k[4], 'cx': info.k[2], 'cy': info.k[5], 'width': info.width, 'height': info.height, 'mtx': np.array(info.k).reshape(3, 3), 'dist': np.array(info.d)}

    def on_joint_state_update(self, msg: JointState):
        self.joint_state = np.array(msg.position)

    def on_right_arm_kinematics_update(self, msg: ARMstrongKinematics):
        pos = ros_pos_to_np_pos(msg.pose.position)
        rot = ros_quat_to_rotation(msg.pose.orientation)
        self.r_pose = (pos, rot)

    def on_head_mouse_event(self, event: int, x: int, y: int, flags: int, params=None) -> None:
        if self.head_bolts_detection_info is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.head_bolts_detection_info.xyxy):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    target = self.set_target(
                        box_id=self.head_bolts_detection_info.box_id[i],
                        group_id=self.head_bolts_detection_info.group_id[i] 
                            if self.head_bolts_detection_info.group_id is not None 
                            else None
                        )
                    self.target_pose = self.get_target_pose(target, tgt_offset=[-0.50, 0, 0.02])
                    if self.target_pose is not None:
                        self.follow_target(self.target_pose)
                    return
                
    def on_hand_mouse_event(self, event: int, x: int, y: int, flags: int, params=None) -> None:
        if self.hand_bolts_detection_info is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.hand_bolts_detection_info.xyxy):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target = self.make_bolt_name(self.hand_bolts_detection_info.box_id[i], self.hand_bolts_detection_info.group_id[i], 'hand')
                    self.docking_status = 'DOCKING'
                    self.docking_pose = self.get_transform('origin', 'r_link6')
                    # self.target_transform = 

    def on_head_draw_timer(self):
        rgb = self.head_rgb
        if rgb is None:
            return
            
        current_time = self.get_clock().now()
        if self.head_prev_time is None:
            self.head_prev_time = current_time
        self.head_frame_time.append((current_time - self.head_prev_time).nanoseconds)
        if len(self.head_frame_time) > 100:
            self.head_frame_time.pop(0)
        self.head_fps = 1 / np.mean(self.head_frame_time) * 1e9
        self.head_prev_time = current_time
        
        # Update target pose if target is set
        if self.target is not None and 'head' in self.target:
            self.target_pose = self.get_target_pose(self.target, tgt_offset=[-0.50, 0, 0.02])
        
        # # Process depth if available
        # if self.head_depth is not None:
        #     try:
        #         depth = self.head_depth.copy()
        #         min_ = cv2.getTrackbarPos('min', 'head_depth') / 1000
        #         max_ = cv2.getTrackbarPos('max', 'head_depth') / 1000
                
        #         depth = np.clip(depth, min_, max_)
        #         depth_norm = np.zeros_like(depth, dtype=np.uint8)
        #         cv2.normalize(depth, depth_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #         depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        #         if self.head_bolts_detection_info is not None:
        #             depth = self.head_detector.draw(depth, self.head_bolts_detection_info, self.head_bolts)
                
        #         cv2.imshow('head_depth', depth)
        #     except Exception as e:
        #         self.get_logger().error(f"Error processing depth image: {str(e)}")

        if self.head_bolts_detection_info is not None:
            rgb = self.head_detector.draw(rgb, self.head_bolts_detection_info, self.head_bolts)

        # Display performance metrics and status
        y_offset = 30
        line_height = 30
        
        # Performance metrics
        rgb = cv2.putText(rgb, f'FPS: {self.head_fps:.2f}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Detection status
        rgb = cv2.putText(rgb, f'Detection: {"ON" if self.is_head_update_detection else "OFF"}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.is_head_update_detection else (0, 0, 255), 2)
        y_offset += line_height
        
        # Target status
        rgb = cv2.putText(rgb, f'Target: {self.target}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Planning status
        status_color = (0, 255, 0)  # Green for IDLE/COMPLETED
        if self.armstrong_plan_status == ARMstrongPlanStatus.PLANNING:
            status_color = (255, 255, 0)  # Yellow for PLANNING
        elif self.armstrong_plan_status == ARMstrongPlanStatus.EXECUTING:
            status_color = (0, 255, 255)  # Cyan for EXECUTING
        elif self.armstrong_plan_status == ARMstrongPlanStatus.FAILED:
            status_color = (0, 0, 255)  # Red for FAILED
            
        status_text = f'Planning: {self.planning_status_map.get(self.armstrong_plan_status, "UNKNOWN")}'
        rgb = cv2.putText(rgb, status_text, (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        cv2.imshow('head_rgb', rgb)

        k = cv2.waitKey(1) % 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
        elif k == ord('f'):
            self.toggle_fix_pose()
        elif k == ord('r'):
            self.armstrong_plan_status = ARMstrongPlanStatus.IDLE

    def on_hand_draw_timer(self):
        rgb = self.hand_rgb
        if rgb is None:
            return
        if len(self.hand_rgb_intrinsics.keys()) == 0:
            return
        
        current_time = self.get_clock().now()
        if self.hand_prev_time is None:
            self.hand_prev_time = current_time
        self.hand_frame_time.append((current_time - self.hand_prev_time).nanoseconds)
        if len(self.hand_frame_time) > 100:
            self.hand_frame_time.pop(0)
        self.hand_fps = 1 / np.mean(self.hand_frame_time) * 1e9
        self.hand_prev_time = current_time

        if self.hand_bolts_detection_info is not None:
            rgb = self.hand_detector.draw(rgb, self.hand_bolts_detection_info, self.hand_bolts)

        tolerance = 50
        center_px = (self.hand_rgb_intrinsics['width'] // 2, self.hand_rgb_intrinsics['height'] // 2)
        box = [center_px[0] - tolerance, center_px[1] - tolerance, center_px[0] + tolerance, center_px[1] + tolerance]
        rgb = cv2.rectangle(rgb, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        if (self.target is not None 
            and 'hand' in self.target 
            and self.docking_status != 'DOCKED'
            and self.hand_bolts_detection_info is not None
            and len(self.hand_bolts_detection_info.center_pixel) > 0
            and len(self.hand_bolts_detection_info.box_id) > 0):
            target_id = int(self.target.split('-')[-2])
            target_indices = (self.hand_bolts_detection_info.box_id == target_id).nonzero()[0]
            if len(target_indices) > 0:
                target_idx = target_indices[0]
                target_point = self.hand_bolts_detection_info.center_pixel[target_idx]
                z_dist =  self.hand_bolts_detection_info.center_point[target_idx][-1]
                self.target_transform = self.perform_docking(target_point, box, z_dist, target_z=0.05)
                rgb = cv2.putText(rgb, f'target_point: {target_point}', (10, 150), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                rgb = cv2.putText(rgb, f'center_point: {self.hand_bolts_detection_info.center_point[target_idx]}', (10, 180), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                rgb = cv2.putText(rgb, f'center_pixel: {self.hand_bolts_detection_info.center_pixel[target_idx]}', (10, 210), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                rgb = cv2.putText(rgb, f'z_dist: {z_dist}', (10, 240), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        rgb = cv2.putText(rgb, f'FPS: {self.hand_fps:.2f}', (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rgb = cv2.putText(rgb, f'Target: {self.target}', (10, 60), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rgb = cv2.putText(rgb, f'Docking Status: {self.docking_status}', (10, 90), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rgb = cv2.putText(rgb, f'Plan Status: {self.planning_status_map.get(self.armstrong_plan_status, "UNKNOWN")}', (10, 120), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('hand_rgb', rgb)

        k = cv2.waitKey(1) % 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()

    def on_armstrong_plan_status_update(self, msg: ARMstrongPlanStatus):
        self.armstrong_plan_status = msg.status
        
    def set_target(self, box_id, group_id, prefix: Literal['head', 'hand'] = 'head') -> str:
        self.target = self.make_bolt_name(box_id, group_id, prefix)
        self.get_logger().info(f"Target set to: {self.target}")
        return self.target
    
    def perform_docking(self, target_point: List | Tuple[int, int], box: List | Tuple[int, int, int, int], z_dist: float, tolerance: int = 50, target_z: float = 0.1):
        assert self.docking_pose is not None
        if self.armstrong_plan_status not in [ARMstrongPlanStatus.IDLE, ARMstrongPlanStatus.COMPLETED, ARMstrongPlanStatus.FAILED]:
            return
        t_x, t_y = target_point
        center_px = (self.hand_rgb_intrinsics['width'] / 2, self.hand_rgb_intrinsics['height'] / 2)

        if box[0] <= t_x <= box[2] and box[1] <= t_y <= box[3]:
            if z_dist < target_z:
                self.docking_status = 'DOCKED'
                return
            self.docking_pose.translation.x += 0.01
        else:
            if t_x < box[0]:
                self.docking_pose.translation.y += 0.01
            elif t_x > box[2]:
                self.docking_pose.translation.y -= 0.01
            if t_y < box[1]:
                self.docking_pose.translation.z += 0.01
            elif t_y > box[3]:
                self.docking_pose.translation.z -= 0.01 

        self.get_logger().info(f"Docking pose: {self.docking_pose.translation.x}, {self.docking_pose.translation.y}, {self.docking_pose.translation.z}")
        docking_pose = Pose()
        docking_pose.position.x = self.docking_pose.translation.x
        docking_pose.position.y = self.docking_pose.translation.y
        docking_pose.position.z = self.docking_pose.translation.z
        docking_pose.orientation = self.docking_pose.rotation
        self.get_logger().info(f"Target point: {t_x}, {t_y}")
        self.get_logger().info(f"Center Pixel: {center_px[0]}, {center_px[1]}")
        msg = self.build_armstrong_plan_request(
            group_name='right_arm',
            link_name='r_link6',
            target_pose=docking_pose,
            wait_after_complete=1.0,
            vel_scale=0.05,
        )
        self.armstrong_plan_pub.publish(msg)
    
    def perform_transform(self, source_pose: Tuple[np.ndarray, Rotation], target_transform: Tuple[np.ndarray, Rotation]):
        source_pos = source_pose[0]
        source_rot = source_pose[1]

        target_tls = target_transform[0]
        target_rot = target_transform[1]

        return_pos = source_rot.apply(target_tls) + source_pos
        return_rot = source_rot * target_rot    

        return (return_pos, return_rot)
    
    def publish_transform(self, target: Tuple[str, np.ndarray, Rotation], source_frame: str) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = source_frame
        tf.child_frame_id = source_frame[:4] + '-' + target[0]
        tf.transform.translation = list_to_ros_vec3(target[1])
        tf.transform.rotation = rot_to_ros_quat(target[2])

        self.tf_broadcaster.sendTransform(tf)

    def follow_target(self, target_pose: Pose) -> None:
        msg = self.build_armstrong_plan_request(
            group_name='right_arm',
            link_name='r_link6',
            target_pose=target_pose,
            wait_after_complete=0.1,
            vel_scale=0.2,
        )
        if self.armstrong_plan_status in [ARMstrongPlanStatus.IDLE, ARMstrongPlanStatus.COMPLETED, ARMstrongPlanStatus.FAILED]:
            self.armstrong_plan_pub.publish(msg)

    def make_bolt_name(self, box_id, group_id, prefix: Literal['head', 'hand'] = 'head'):
        return '-'.join(map(str, [prefix, group_id, box_id, 'bolt'] if group_id is not None else ['unknown', box_id, 'bolt']))

    def get_target_pose(self, bolt_to_follow: str, tgt_offset = None, link_offset = None):
        tf = self.get_transform('origin', bolt_to_follow)
        if tf is None:
            return
        
        pos = ros_pos_to_np_pos(tf.translation)
        rot = ros_quat_to_rotation(tf.rotation)
        rot *= Rotation.from_rotvec([0, -90, 0], True)
        rot *= Rotation.from_rotvec([-90, 0, 0], True)

        euler = rot.as_euler('xyz')
        euler[0] = 0
        euler[1] = 0
        rot = Rotation.from_euler('xyz', euler)

        if tgt_offset is not None:
            pos += rot.apply(tgt_offset)
        if link_offset is not None:
            pos += link_offset

        tf = TransformStamped()
        tf.header.frame_id = 'origin'
        tf.child_frame_id = 'target'
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.transform.translation = list_to_ros_vec3(pos)
        tf.transform.rotation = rot_to_ros_quat(rot)
        self.tf_broadcaster.sendTransform(tf)

        target_pose = Pose()
        target_pose.position = list_to_ros_point(pos)
        target_pose.orientation = rot_to_ros_quat(rot)
        return target_pose

    def get_transform(self, origin, target) -> Transform | None:
        now = self.get_clock().now()
        if not self.tf_buffer.can_transform(origin, target, Time()):
            return
        
        ts = self.tf_buffer.lookup_transform(origin, target, Time())
        
        return ts.transform     
    
    def toggle_fix_pose(self):
        self.is_head_update_detection = not self.is_head_update_detection

    def toggle_camera_location(self, location: Optional[Literal['head', 'hand']] = None):
        if location is None:
            self.is_camera_head = not self.is_camera_head
        else:
            self.is_camera_head = location == 'head'

    def build_armstrong_plan_request(
        self,
        group_name: str,
        frame: str = 'origin',
        planning_pipeline: str = 'pilz_industrial_motion_planner',
        planner_id: str = 'PTP',
        link_name: Optional[str] = None,
        target_pose: Optional[Pose] = None,
        target_transform: Optional[Transform] = None,
        target_joint: Optional[JointState] = None,
        vel_scale: float = 0.5,
        acc_scale: Optional[float] = None,
        offset: Optional[list[float]] = None,
        ori_tol: Optional[list[float]] = None,
        is_lin: bool = False,
        is_seq: bool = False,
        is_local: bool = False,
        ik_only: bool = False,
        trigger: int = 500,
        wait_after_complete: float = 0.0
        ) -> ARMstrongPlanRequest:

        msg = ARMstrongPlanRequest()
        msg.frame = frame
        msg.group_name = group_name
        msg.planning_pipeline = planning_pipeline
        msg.planner_id = planner_id
        if link_name is not None:
            msg.link_name = link_name
        assert target_pose or target_transform or target_joint, "At least one of target_pose, target_transform, or target_joint must be provided."
        if target_pose is not None:
            msg.target_pose = target_pose
            msg.command_type = ARMstrongPlanRequest.POSE
        if target_transform is not None:
            msg.target_transform = target_transform
            msg.command_type = ARMstrongPlanRequest.TRANSFORM
        if target_joint is not None:
            msg.target_joint = target_joint
            msg.command_type = ARMstrongPlanRequest.JOINT
        msg.vel_scale = vel_scale
        if acc_scale is None:
            msg.acc_scale = vel_scale
        else:
            msg.acc_scale = acc_scale
        if offset is not None:
            msg.offset = offset
        msg.is_lin = is_lin
        if is_lin and planning_pipeline == 'pilz_industrial_motion_planner':
            msg.planner_id = 'LIN'

        msg.is_seq = is_seq
        msg.trigger = trigger
        msg.is_local = is_local
        msg.ik_only = ik_only

        if ori_tol is not None:
            msg.orientation_tolerance = ori_tol

        msg.wait_after_complete = wait_after_complete

        return msg


def main():
    rclpy.init()
    node = BoltFastenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()


