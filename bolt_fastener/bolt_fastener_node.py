from typing import Tuple, Optional
from copy import deepcopy

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

        self.detector = BoltDetector(model_path=model)
        self.processor = PointCloudProcessor()

        tcp_tunnel_prefix = '/tcp_tunnel_client' if use_tcp_tunnel else ''
        camera_namespace = '/camera/camera'

        self.target = None
        self.target_pose = None
        self.depth_intrinsics = {}
        self.confidence_threshold = 0.
        self.detection: DetectionResult | None = None
        self.bolts = []
        self.bolts_detection_info: DetectionResult | None = None
        self.joint_state: np.ndarray | None = None
        self.r_pose: Tuple[np.ndarray, Rotation] = (np.zeros(3), Rotation.identity())
        self.armstrong_plan_status = ARMstrongPlanStatus.IDLE

        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.is_update_detection: bool = True

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        head_cam_transform = TransformStamped()
        head_cam_transform.header.stamp = self.get_clock().now().to_msg()
        head_cam_transform.header.frame_id = 'origin'
        head_cam_transform.child_frame_id = 'camera_link'
        head_cam_transform.transform.translation = list_to_ros_vec3(np.array([-75.31, 0, 531.14]) / 1000)
        head_cam_transform.transform.rotation = rot_to_ros_quat(Rotation.from_rotvec([0, 30*np.pi/180, 0]))
        
        # hand_cam_transform = TransformStamped()
        # hand_cam_transform.header.stamp = self.get_clock().now().to_msg()
        # hand_cam_transform.header.frame_id = 'origin'
        # hand_cam_transform.child_frame_id = 'hand_camera_link'
        # hand_cam_transform.tr

        tf_static_broadcaster.sendTransform([head_cam_transform,])

        self.rgb_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + '/color/image_raw/compressed',
            callback=self.on_image_update,
            qos_profile=qos_profile_system_default,
            )
        self.depth_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=tcp_tunnel_prefix + camera_namespace + '/aligned_depth_to_color/image_raw/compressedDepth',
            callback=self.on_depth_update,
            qos_profile=qos_profile_system_default,
            )
        self.depth_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=camera_namespace + "/aligned_depth_to_color/camera_info",
            callback=self.on_depth_info_update,
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
        cv2.namedWindow('depth')
        cv2.createTrackbar('min', 'depth', 0, 10000, lambda x: None)
        cv2.createTrackbar("max", 'depth', 0, 10000, lambda x: None)

        cv2.setTrackbarPos('max', 'depth', 10000)
        
        cv2.namedWindow('rgb')
        cv2.createTrackbar('Contrast', 'rgb', 0, 100, lambda x: None)
        cv2.createTrackbar('Brightness', 'rgb', 0, 100, lambda x: None)
        cv2.createTrackbar("Confidence", 'rgb', 0, 100, lambda x: None)

        cv2.setTrackbarPos('Contrast', 'rgb', 50)
        cv2.setTrackbarPos('Brightness', 'rgb', 50)
        cv2.setTrackbarPos('Confidence', 'rgb', 50)
        cv2.setMouseCallback('rgb', self.on_mouse_event)

        
        self.draw_timer = self.create_timer(1/30, self.on_draw_timer)
        self.prev_time = self.get_clock().now()
        self.frame_time = []
        self.fps = 0
        self.rolling_avg_fps = 0.0
        self.alpha = 0.1  # Smoothing factor for rolling average
        self.planning_status_map = {
            ARMstrongPlanStatus.IDLE: "IDLE",
            ARMstrongPlanStatus.PLANNING: "PLANNING",
            ARMstrongPlanStatus.EXECUTING: "EXECUTING",
            ARMstrongPlanStatus.COMPLETED: "COMPLETED",
            ARMstrongPlanStatus.FAILED: "FAILED"
        }
    
    def on_image_update(self, msg: CompressedImage):
        contrast = cv2.getTrackbarPos('Contrast', 'rgb')
        brightness = cv2.getTrackbarPos("Brightness", 'rgb')
        self.confidence_threshold = cv2.getTrackbarPos("Confidence", 'rgb') / 100.0
        
        str_msg = msg.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                         dtype=np.uint8, buffer=msg.data)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = image.astype(np.uint8)
        
        img = cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)
        self.rgb = img
        
        res = self.detector.track(img)
        if res is not None:
            self.detection = res
        
    def on_depth_update(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        depth = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
        self.depth = depth.astype(np.float32) / 1000

        if (self.detection is not None 
            and self.depth_intrinsics != {} 
            and self.depth is not None
            and self.is_update_detection
            ):
            res = self.processor.estimate_bolt_pose_pipeline(
                depth=self.depth,
                detection=self.detection,
                conf=self.confidence_threshold,
                crop_by_roi=True,
                radius=0.03,
                force_single_estimation=False)
            if res is None:
                return
            self.bolts, self.bolts_detection_info = res
        for b in self.bolts:
            self.publish_transform(b, 'camera_color_optical_frame')

    def on_depth_info_update(self, info: CameraInfo):
        depth_intrinsics = {'fx': info.k[0], 'fy': info.k[4], 'cx': info.k[2], 'cy': info.k[5], 'width': info.width, 'height': info.height, 'mtx': np.array(info.k).reshape(3, 3), 'dist': np.array(info.d)}
        self.depth_intrinsics = depth_intrinsics
        self.processor.set_depth_intrinsics(depth_intrinsics)
        self.detector.cam_info = depth_intrinsics

    def on_joint_state_update(self, msg: JointState):
        self.joint_state = np.array(msg.position)

    def on_right_arm_kinematics_update(self, msg: ARMstrongKinematics):
        pos = ros_pos_to_np_pos(msg.pose.position)
        rot = ros_quat_to_rotation(msg.pose.orientation)
        self.r_pose = (pos, rot)

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, params=None) -> None:
        if self.bolts_detection_info is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.bolts_detection_info.xyxy):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    target = self.set_target(
                        box_id=self.bolts_detection_info.box_id[i],
                        group_id=self.bolts_detection_info.group_id[i] 
                            if self.bolts_detection_info.group_id is not None 
                            else None
                        )
                    self.target_pose = self.get_target_pose(target, tgt_offset=[-0.30, 0, 0.02])
                    if self.target_pose is not None:
                        self.follow_target(self.target_pose)
                    return

    def on_draw_timer(self):
        rgb = self.rgb
        if rgb is None:
            return
            
        current_time = self.get_clock().now()
        if self.prev_time is None:
            self.prev_time = current_time
        self.frame_time.append((current_time - self.prev_time).nanoseconds)
        if len(self.frame_time) > 100:
            self.frame_time.pop(0)
        self.fps = 1 / np.mean(self.frame_time) * 1e9
        self.prev_time = current_time
        
        # Update target pose if target is set
        if self.target is not None:
            self.target_pose = self.get_target_pose(self.target, tgt_offset=[-0.30, 0, 0.02])
        
        # Process depth if available
        if self.depth is not None:
            try:
                depth = self.depth.copy()
                min_ = cv2.getTrackbarPos('min', 'depth') / 1000
                max_ = cv2.getTrackbarPos('max', 'depth') / 1000
                
                depth = np.clip(depth, min_, max_)
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
                cv2.normalize(depth, depth_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                if self.bolts_detection_info is not None:
                    depth = self.detector.draw(depth, self.bolts_detection_info, self.bolts)
                
                cv2.imshow('depth', depth)
            except Exception as e:
                self.get_logger().error(f"Error processing depth image: {str(e)}")

        if self.bolts_detection_info is not None:
            rgb = self.detector.draw(rgb, self.bolts_detection_info, self.bolts)

        # Display performance metrics and status
        y_offset = 30
        line_height = 30
        
        # Performance metrics
        rgb = cv2.putText(rgb, f'FPS: {self.fps:.2f}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Detection status
        rgb = cv2.putText(rgb, f'Detection: {"ON" if self.is_update_detection else "OFF"}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.is_update_detection else (0, 0, 255), 2)
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

        cv2.imshow('rgb', rgb)

        k = cv2.waitKey(1) % 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
        elif k == ord('f'):
            self.toggle_fix_pose()
        
    def on_armstrong_plan_status_update(self, msg: ARMstrongPlanStatus):
        self.armstrong_plan_status = msg.status
        
    def set_target(self, box_id, group_id) -> str:
        self.target = '-'.join(map(str, [group_id, box_id, 'bolt'] if group_id is not None else ['unknown', box_id, 'bolt']))
        self.get_logger().info(f"Target set to: {self.target}")
        return self.target
    
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
        tf.child_frame_id = target[0]
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

    def make_bolt_name(self, box_id, group_id):
        return '-'.join(map(str, [group_id, box_id, 'bolt'] if group_id is not None else ['unknown', box_id, 'bolt']))

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
        self.is_update_detection = not self.is_update_detection

    
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


