from typing import Tuple, Optional, Literal, List, Callable
import subprocess
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_system_default
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from ultralytics import SAM

from ament_index_python import get_package_share_directory

from tf2_ros import Buffer, TransformBroadcaster, TransformListener, StaticTransformBroadcaster


from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, Imu
from geometry_msgs.msg import TransformStamped, Pose, Transform

from ass_msgs.msg import ARMstrongKinematics, ARMstrongPlanRequest, ARMstrongPlanStatus, DXLCommand

from armstrong_py.conversions import ros_pos_to_np_pos, ros_quat_to_rotation, list_to_ros_vec3, list_to_ros_point, rot_to_ros_quat, transform_to_pose, pose_to_transform

from bolt_fastener.bolt_detector import BoltDetector, DetectionResult
from bolt_fastener.point_cloud_processor import PointCloudProcessor
from bolt_fastener.insert_status import InsertStatus


class BoltFastenerNode(Node):
    def __init__(self, name='bolt_fastener', use_tcp_tunnel=True):
        super().__init__(name)

        # Initialize detectors and processors
        self._initialize_detectors()

        # Setup camera configuration
        self._setup_camera_config(use_tcp_tunnel)

        # Initialize state variables
        self._initialize_state_variables()

        # Setup TF
        self._setup_tf()

        # Setup subscribers
        self._setup_subscribers()

        # Setup publishers
        self._setup_publishers()

        # Setup clients
        self._setup_clients()

        # Setup visualization
        self._setup_visualization()

        # Setup loops
        self._setup_loops()

        # Setup status maps
        self._setup_status_maps()

    def _initialize_detectors(self):
        """Initialize bolt detectors and point cloud processors"""
        model = get_package_share_directory('bolt_fastener') + '/best.pt'
        self.head_detector = BoltDetector(model_path=model)
        self.head_processor = PointCloudProcessor()
        self.hand_detector = BoltDetector(model_path=model)
        self.hand_processor = PointCloudProcessor()
        self.sam = SAM('/home/arm/ros2ws/src/bolt_fastener/bolt_fastener/sam2.1_s.pt')

    def _setup_camera_config(self, use_tcp_tunnel):
        """Setup camera configuration and prefixes"""
        self.tcp_tunnel_prefix = '/tcp_tunnel_client' if use_tcp_tunnel else ''
        self.camera_namespace = '/m4_camera'
        self.head_camera_name = '/head_camera'
        self.hand_camera_name = '/hand_camera'

    def _initialize_state_variables(self):
        """Initialize all state variables"""
        # Tracking and fastening variables
        self.target = None
        self.target_pose = None
        self._status: Literal['IDLE', 'ALIGNING', 'ALIGNED', 'TUNED', 'DOCKING', 'DOCKED', 'INSERTING', 'INSERTED', 'FAILED', 'RETRYING'] = 'IDLE'
        self.docking_pose: Pose | None = None
        self.align_offset = [0.4, 0, 0.05]  # Offset for alignment
        self.insert_translation = [0.0, 0.0, 0.0]  # Initialize insert translation vector
        self.tolerance = 25  # px
        self.colored_mask: np.ndarray | None = None
        self.retry_count = 0

        # Head camera state
        self.head_depth_intrinsics = {}
        self.head_rgb_intrinsics = {}
        self.head_confidence_threshold = 0.
        self.head_detection: DetectionResult | None = None
        self.head_bolts = []
        self.head_bolts_detection_info: DetectionResult | None = None
        self.head_rgb: np.ndarray | None = None
        self.head_depth: np.ndarray | None = None
        self.is_head_update_detection: bool = True

        # Hand camera state
        self.hand_bolts = []
        self.hand_confidence_threshold = 0.
        self.hand_detection: DetectionResult | None = None
        self.hand_bolts_detection_info: DetectionResult | None = None
        self.hand_rgb: np.ndarray | None = None
        self.hand_depth: np.ndarray | None = None
        self.hand_rgb_intrinsics = {}
        self.hand_depth_intrinsics = {}
        self.is_hand_update_detection: bool = True

        # Robot state
        self.joint_state: np.ndarray | None = None
        self.armstrong_plan_status = ARMstrongPlanStatus.IDLE
        self.is_camera_head: bool = True
        self.box_scale = 0.
        self.projected_bolt_box_area = None
        self.imu_ori = Rotation.identity()  # imu orientation wrt. ground (gravity)

        self.reentrant_group = ReentrantCallbackGroup()
        self.mutex_group = MutuallyExclusiveCallbackGroup()

    def _setup_tf(self):
        """Setup TF listeners and broadcasters"""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Setup static transforms for cameras
        self._setup_camera_transforms(tf_static_broadcaster)

    def _setup_camera_transforms(self, tf_static_broadcaster):
        """Setup static transforms for head and hand cameras"""
        # Head camera transform
        head_cam_transform = TransformStamped()
        head_cam_transform.header.stamp = self.get_clock().now().to_msg()
        head_cam_transform.header.frame_id = 'origin'
        head_cam_transform.child_frame_id = 'head_camera_link'
        head_cam_transform.transform.translation = list_to_ros_vec3(np.array([-75.31, 0+47.5, 531.14]) / 1000)
        head_cam_transform.transform.rotation = rot_to_ros_quat(Rotation.from_rotvec([0, 30*np.pi/180, 0]))

        # Hand camera transform
        hand_cam_transform = TransformStamped()
        hand_cam_transform.header.stamp = self.get_clock().now().to_msg()
        hand_cam_transform.header.frame_id = 'r_link6'
        hand_cam_transform.child_frame_id = 'hand_camera_link'
        hand_cam_transform.transform.translation = list_to_ros_vec3(np.array([228.01, -35.5-17.5, 46.]) / 1000)
        hand_cam_transform.transform.rotation = rot_to_ros_quat(Rotation.identity())

        tf_static_broadcaster.sendTransform(head_cam_transform)
        tf_static_broadcaster.sendTransform(hand_cam_transform)

    def _setup_subscribers(self):
        """Setup all subscribers"""
        # Head camera subscribers
        self._setup_head_camera_subscribers()

        # Hand camera subscribers
        self._setup_hand_camera_subscribers()

        # Robot state subscribers
        self._setup_robot_subscribers()

    def _setup_head_camera_subscribers(self):
        """Setup subscribers for head camera"""
        self.rgb_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=self.tcp_tunnel_prefix + self.camera_namespace + self.head_camera_name + '/color/image_raw/compressed',
            callback=self.on_head_image_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.depth_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=self.tcp_tunnel_prefix + self.camera_namespace + self.head_camera_name + '/aligned_depth_to_color/image_raw/compressedDepth',
            callback=self.on_head_depth_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.depth_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.camera_namespace + self.head_camera_name + "/aligned_depth_to_color/camera_info",
            callback=self.on_head_depth_info_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.head_rgb_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.camera_namespace + self.head_camera_name + "/color/camera_info",
            callback=self.on_head_rgb_info_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )

    def _setup_hand_camera_subscribers(self):
        """Setup subscribers for hand camera"""
        self.hand_rgb_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=self.tcp_tunnel_prefix + self.camera_namespace + self.hand_camera_name + '/color/image_raw/compressed',
            callback=self.on_hand_image_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.hand_depth_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic=self.tcp_tunnel_prefix + self.camera_namespace + self.hand_camera_name + '/aligned_depth_to_color/image_raw/compressedDepth',
            callback=self.on_hand_depth_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.hand_depth_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.camera_namespace + self.hand_camera_name + "/aligned_depth_to_color/camera_info",
            callback=self.on_hand_depth_info_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.hand_rgb_info_sub = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.camera_namespace + self.hand_camera_name + "/color/camera_info",
            callback=self.on_hand_rgb_info_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.hand_imu_sub = self.create_subscription(
            msg_type=Imu,
            topic="/imu/data",
            callback=self.on_hand_imu_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )

    def _setup_robot_subscribers(self):
        """Setup subscribers for robot state"""
        self.joint_state_sub = self.create_subscription(
            msg_type=JointState,
            topic='joint_states',
            callback=self.on_joint_state_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )
        self.armstrong_plan_status_sub = self.create_subscription(
            msg_type=ARMstrongPlanStatus,
            topic="armstrong_plan_status",
            callback=self.on_armstrong_plan_status_update,
            callback_group=self.reentrant_group,
            qos_profile=qos_profile_system_default,
        )

    def _setup_publishers(self):
        """Setup all publishers"""
        self.armstrong_plan_pub = self.create_publisher(
            msg_type=ARMstrongPlanRequest,
            topic='armstrong_plan_request',
            qos_profile=qos_profile_system_default,
            callback_group=self.reentrant_group,
        )
        self.dxl_command_pub = self.create_publisher(
            msg_type=DXLCommand,
            topic='dxl_command',
            qos_profile=qos_profile_system_default,
            callback_group=self.reentrant_group,
        )

    def _setup_clients(self):
        """Setup all clients"""
        def get_param():
            res = subprocess.run(['ros2', 'param', 'get', 'backass', 'offset.R6'], capture_output=True, text=True)
            return float(res.stdout.split()[-1])
        def set_param(value):
            subprocess.run(['ros2', 'param', 'set', 'backass', 'offset.R6', str(value)])
        self.get_param: Callable[[], float] = get_param
        self.set_param: Callable[[float], None] = set_param


    def _setup_visualization(self):
        """Setup visualization windows and trackbars"""
        # Head camera visualization
        cv2.namedWindow('head_rgb')
        cv2.createTrackbar('Contrast', 'head_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar('Brightness', 'head_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar("Confidence", 'head_rgb', 0, 100, lambda x: None)
        cv2.setTrackbarPos('Contrast', 'head_rgb', 50)
        cv2.setTrackbarPos('Brightness', 'head_rgb', 50)
        cv2.setTrackbarPos('Confidence', 'head_rgb', 50)
        cv2.setMouseCallback('head_rgb', self.on_head_mouse_event)

        # Hand camera visualization
        cv2.namedWindow('hand_rgb')
        cv2.createTrackbar('Contrast', 'hand_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar('Brightness', 'hand_rgb', 0, 100, lambda x: None)
        cv2.createTrackbar("Confidence", 'hand_rgb', 0, 100, lambda x: None)
        cv2.setTrackbarPos('Contrast', 'hand_rgb', 50)
        cv2.setTrackbarPos('Brightness', 'hand_rgb', 50)
        cv2.setTrackbarPos('Confidence', 'hand_rgb', 50)
        cv2.setMouseCallback('hand_rgb', self.on_hand_mouse_event)

        # Hand depth visualization
        cv2.namedWindow('hand_depth')
        cv2.createTrackbar('Min Depth', 'hand_depth', 0, 1000, lambda x: None)
        cv2.createTrackbar('Max Depth', 'hand_depth', 0, 1000, lambda x: None)
        cv2.setTrackbarPos('Min Depth', 'hand_depth', 0)
        cv2.setTrackbarPos('Max Depth', 'hand_depth', 1000)

    def _setup_loops(self):
        """Setup loops for visualization and updates"""
        # Head camera timer
        self.head_draw_timer = self.create_timer(1/30, self.on_head_draw_timer, callback_group=self.reentrant_group)
        self.head_prev_time = self.get_clock().now()
        self.head_frame_time = []
        self.head_fps = 0.

        # Hand camera timer
        self.hand_draw_timer = self.create_timer(1/30, self.on_hand_draw_timer, callback_group=self.reentrant_group)
        self.hand_prev_time = self.get_clock().now()
        self.hand_frame_time = []
        self.hand_fps = 0.

        self.future_done = False

        # Control timer for robot motion
        self.control_timer = self.create_timer(1/5, self.on_control_loop, callback_group=self.mutex_group)
        # self.control_loop = threading.Thread(target=self.on_control_loop)
        # self.control_loop.start()


    def _setup_status_maps(self):
        """Setup status mapping dictionaries"""
        self.planning_status_map = {
            ARMstrongPlanStatus.IDLE: "IDLE",
            ARMstrongPlanStatus.PLANNING: "PLANNING",
            ARMstrongPlanStatus.EXECUTING: "EXECUTING",
            ARMstrongPlanStatus.COMPLETED: "COMPLETED",
            ARMstrongPlanStatus.FAILED: "FAILED"
        }

        M32_insert_status = InsertStatus(
            approach=(0.003, -0.01, 0.072),
            insert=(0.035, 0, 0),
            fasten=(0, 0, 0),
            retract=(-0.035, 0, 0),
            full_retract=tuple(np.deg2rad((-25., 0., 0., 25., 0., 50.)))
        )

        self.insert_status = M32_insert_status

    def on_head_image_update(self, msg: CompressedImage):
        """Process head camera RGB image update"""
        try:
            if self.head_rgb_intrinsics == {}:
                return

            # Get trackbar positions with error checking
            try:
                contrast = cv2.getTrackbarPos('Contrast', 'head_rgb')
                brightness = cv2.getTrackbarPos("Brightness", 'head_rgb')
                self.head_confidence_threshold = cv2.getTrackbarPos("Confidence", 'head_rgb') / 100.0
            except Exception as e:
                self.get_logger().warning(f"Failed to read trackbar values: {str(e)}")
                contrast, brightness = 50, 50
                self.head_confidence_threshold = 0.5

            # Decode and process image
            try:
                str_msg = msg.data
                buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=msg.data)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                if image is None:
                    self.get_logger().warning("Failed to decode head camera image")
                    return

                img = image.astype(np.uint8)
                img = cv2.undistort(img, self.head_rgb_intrinsics['mtx'], self.head_rgb_intrinsics['dist'])
                # Apply contrast and brightness
                img = cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)
                self.head_rgb = img

                # Update detection
                res = self.head_detector.track(img)
                if res is not None:
                    self.head_detection = res
            except cv2.error as e:
                self.get_logger().error(f"OpenCV error processing head image: {str(e)}")
            except ValueError as e:
                self.get_logger().error(f"Value error processing head image: {str(e)}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error processing head image: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Critical error in head image processing: {str(e)}")

    def on_hand_image_update(self, msg: CompressedImage):
        """Process hand camera RGB image update"""
        if self.hand_rgb_intrinsics == {}:
            return
        contrast = cv2.getTrackbarPos('Contrast', 'hand_rgb')
        brightness = cv2.getTrackbarPos("Brightness", 'hand_rgb')
        self.hand_confidence_threshold = cv2.getTrackbarPos("Confidence", 'hand_rgb') / 100.0

        # Decode and process image
        str_msg = msg.data
        buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=msg.data)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = image.astype(np.uint8)

        # Apply contrast and brightness
        img = cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)
        self.hand_rgb = img

        # Update detection
        res = self.hand_detector.track(img)
        if res is not None:
            self.hand_detection = res

    def on_head_depth_update(self, msg: CompressedImage):
        """Process head camera depth image update"""
        try:
            if self.head_depth_intrinsics == {}:
                return

            # Decode depth image
            try:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                depth = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)

                if depth is None:
                    self.get_logger().warning("Failed to decode head depth image")
                    return

                depth = cv2.undistort(depth, self.head_depth_intrinsics['mtx'], self.head_depth_intrinsics['dist'])
                self.head_depth = depth.astype(np.float32) / 1000  # Convert to meters
            except IndexError as e:
                self.get_logger().error(f"Index error decoding depth image: {str(e)}")
                return
            except cv2.error as e:
                self.get_logger().error(f"OpenCV error processing depth image: {str(e)}")
                return
            except Exception as e:
                self.get_logger().error(f"Error processing depth image: {str(e)}")
                return

            # Process bolt detection if conditions are met
            if (self.head_detection is not None
                and self.head_depth_intrinsics != {}
                and self.head_depth is not None
                and self.is_head_update_detection):

                try:
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
                except ValueError as e:
                    self.get_logger().error(f"Value error in bolt pose estimation: {str(e)}")
                except IndexError as e:
                    self.get_logger().error(f"Index error in bolt pose estimation: {str(e)}")
                except Exception as e:
                    self.get_logger().error(f"Error in bolt pose estimation: {str(e)}")

            # Publish transforms for detected bolts
            try:
                for b in self.head_bolts:
                    self.publish_transform(b, 'head_camera_color_optical_frame')
            except Exception as e:
                self.get_logger().error(f"Error publishing transforms: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Critical error in head depth processing: {str(e)}")

    def on_hand_depth_update(self, msg: CompressedImage):
        """Process hand camera depth image update"""
        try:
            if self.hand_depth_intrinsics == {}:
                return

            # Decode depth image
            try:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                depth = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)

                if depth is None:
                    self.get_logger().warning("Failed to decode hand depth image")
                    return

                self.hand_depth = depth.astype(np.float32) / 1000  # Convert to meters
            except IndexError as e:
                self.get_logger().error(f"Index error decoding hand depth image: {str(e)}")
                return
            except cv2.error as e:
                self.get_logger().error(f"OpenCV error processing hand depth image: {str(e)}")
                return
            except Exception as e:
                self.get_logger().error(f"Error processing hand depth image: {str(e)}")
                return

        except Exception as e:
            self.get_logger().error(f"Critical error in hand depth processing: {str(e)}")

    def on_head_depth_info_update(self, info: CameraInfo):
        """Update head camera depth intrinsics"""
        depth_intrinsics = {
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
            'width': info.width, 'height': info.height,
            'mtx': np.array(info.k).reshape(3, 3),
            'dist': np.array(info.d)
        }

        self.head_depth_intrinsics = depth_intrinsics
        self.head_processor.set_depth_intrinsics(depth_intrinsics)
        self.head_detector.cam_info = depth_intrinsics

    def on_hand_depth_info_update(self, info: CameraInfo):
        """Update hand camera depth intrinsics"""
        self.hand_depth_intrinsics = {
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
            'width': info.width, 'height': info.height,
            'mtx': np.array(info.k).reshape(3, 3),
            'dist': np.array(info.d)
        }
        self.hand_processor.set_depth_intrinsics(self.hand_depth_intrinsics)
        self.hand_detector.cam_info = self.hand_depth_intrinsics

    def on_head_rgb_info_update(self, info: CameraInfo):
        """Update head camera RGB intrinsics"""
        self.head_rgb_intrinsics = {
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
            'width': info.width, 'height': info.height,
            'mtx': np.array(info.k).reshape(3, 3),
            'dist': np.array(info.d)
        }

    def on_hand_rgb_info_update(self, info: CameraInfo):
        """Update hand camera RGB intrinsics"""
        self.hand_rgb_intrinsics = {
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
            'width': info.width, 'height': info.height,
            'mtx': np.array(info.k).reshape(3, 3),
            'dist': np.array(info.d)
        }
    def on_hand_imu_update(self, msg: Imu):
        """Update hand camera IMU"""
        self.imu_ori = Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def on_joint_state_update(self, msg: JointState):
        """Update joint state"""
        self.joint_state = np.array(msg.position)

    def on_armstrong_plan_status_update(self, msg: ARMstrongPlanStatus):
        """Update armstrong plan status"""
        self.armstrong_plan_status = msg.status
        if self.armstrong_plan_status == 'PLANNED' and self.status == 'RETRYING':
            self.retry_count = 0
        if self.armstrong_plan_status == ARMstrongPlanStatus.COMPLETED and self.status == 'ALIGNING':
            self.status = 'ALIGNED'
        elif self.armstrong_plan_status == ARMstrongPlanStatus.FAILED:
            if self.status == 'INSERTING':
                self.insert_status.status = 'IDLE'
                self.status = 'FAILED'
            elif self.status == 'ALIGNING' or self.status == 'RETRYING':
                if self.retry_count > 10:
                    self.status = 'FAILED'
                    self.retry_count = 0
                    return
                self.get_logger().info("Failed to align. Retrying...")
                self.status = 'RETRYING'
                self.retry_align()
            else:
                self.status = 'FAILED'

    def retry_align(self):
        assert self.target_pose is not None
        tgt_roll = 5 * (self.retry_count // 2) * (-1) ** (self.retry_count % 2)
        self.get_logger().info(f"Retrying alignment with roll: {tgt_roll}")
        tgt_rot = ros_quat_to_rotation(self.target_pose.orientation).as_euler('xyz', degrees=True)
        tgt_rot[0] = tgt_roll
        self.target_pose.orientation = rot_to_ros_quat(Rotation.from_euler('xyz', tgt_rot, degrees=True))
        self.follow_target(self.target_pose)
        self.retry_count += 1

    def on_head_mouse_event(self, event: int, x: int, y: int, flags: int, params=None) -> None:
        """Handle mouse events on head camera view"""
        if self.head_bolts_detection_info is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.head_bolts_detection_info.xyxy):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.status = 'ALIGNING'
                    target = self.set_target(
                        box_id=self.head_bolts_detection_info.box_id[i],
                        group_id=self.head_bolts_detection_info.group_id[i]
                            if self.head_bolts_detection_info.group_id is not None
                            else None
                    )
                    self.target_pose = self.get_target_pose(target, tgt_offset=[-0.40, 0, -0.05])
                    if self.target_pose is not None:
                        self.follow_target(self.target_pose)
                    return

    def on_hand_mouse_event(self, event: int, x: int, y: int, flags: int, params=None) -> None:
        """Handle mouse events on hand camera view"""
        if self.hand_detection is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.hand_detection.xyxy):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target = self.make_bolt_name(self.hand_detection.box_id[i], '', 'hand')
                    self.get_logger().info(f"Target set to: {self.target}")
                    self.docking_pose = self.get_transform('origin', 'r_link6')
                    self.status = 'DOCKING'

    def on_head_draw_timer(self) -> None:
        """Update head camera visualization"""
        if self.head_rgb is None:
            return
        rgb = self.head_rgb.copy()

        # Update FPS calculation
        current_time = self.get_clock().now()
        if self.head_prev_time is None:
            self.head_prev_time = current_time
        self.head_frame_time.append((current_time - self.head_prev_time).nanoseconds)
        if len(self.head_frame_time) > 100:
            self.head_frame_time.pop(0)
        self.head_fps = 1 / np.mean(self.head_frame_time) * 1e9
        self.head_prev_time = current_time

        # # Update target pose if target is set
        # if self.target is not None and 'head' in self.target:
        #     self.target_pose = self.get_target_pose(self.target, tgt_offset=[-0.30, 0, -0.05])

        # Draw detections if available
        if self.head_bolts_detection_info is not None:
            rgb = self.head_detector.draw(rgb, self.head_bolts_detection_info, self.head_bolts)

        # Display status information
        self._draw_status_info(rgb)

        # Show image and handle keyboard input
        cv2.imshow('head_rgb', rgb)
        self._handle_keyboard_input()

    def on_hand_draw_timer(self):
        """Update hand camera visualization"""
        if self.hand_rgb is None or len(self.hand_rgb_intrinsics.keys()) == 0:
            return
        rgb = self.hand_rgb.copy()

        # Update FPS calculation
        current_time = self.get_clock().now()
        if self.hand_prev_time is None:
            self.hand_prev_time = current_time
        self.hand_frame_time.append((current_time - self.hand_prev_time).nanoseconds)
        if len(self.hand_frame_time) > 100:
            self.hand_frame_time.pop(0)
        self.hand_fps = 1 / np.mean(self.hand_frame_time) * 1e9
        self.hand_prev_time = current_time

        # Setup projected bolt box if not done
        self._setup_projected_bolt_box()

        # Draw detections if available
        if self.hand_detection is not None:
            rgb = self.hand_detector.draw(rgb, self.hand_detection, self.hand_bolts)

        # Draw target boxes
        self._draw_target_boxes(rgb)

        if self.target is not None and 'hand' in self.target and self.colored_mask is not None and self.status == 'DOCKING':
            rgb = cv2.addWeighted(rgb, 0.5, self.colored_mask, 0.5, 0)

        # Display status information
        self._draw_status_info(rgb)

        # Show image
        cv2.imshow('hand_rgb', rgb)
        cv2.imshow('hand_depth', self.hand_rgb)

    def on_future_done(self, future):
        self.future_done = True

    def on_control_loop(self):
        """Handle robot motion control at a fixed rate"""

        if self.target is None or 'hand' not in self.target:
            return
        if self.status in 'DOCKING':
            self._handle_docking_control_by_sam()
        elif self.status == 'INSERTING' or self.status == 'DOCKED':
            self._handle_inserting_control()
        elif self.status == 'FAILED':
            self.insert_status.status = 'IDLE'

    def _handle_docking_control(self):
        """Handle docking control logic"""
        if (self.target is not None
            and 'hand' in self.target
            and self.status != 'DOCKED'
            and self.hand_detection is not None
            and len(self.hand_detection.center_pixel) > 0
            and len(self.hand_detection.box_id) > 0):

            target_id = int(self.target.split('-')[-2])
            target_indices = (self.hand_detection.box_id == target_id).nonzero()[0]

            if len(target_indices) > 0:
                target_idx = target_indices[0]
                target_point = self.hand_detection.center_pixel[target_idx]
                target_bbox = self.hand_detection.xyxy[target_idx]
                target_bbox_area = np.multiply(*(target_bbox[0:2] - target_bbox[2:]))

                # Update visual servoing target
                center_px = (self.hand_rgb_intrinsics['width'] // 2, self.hand_rgb_intrinsics['height'] // 2)
                box = np.array([center_px[0] - self.tolerance, center_px[1] - self.tolerance,
                              center_px[0] + self.tolerance, center_px[1] + self.tolerance]).astype(np.int32)

                self.perform_docking_by_box(box, target_bbox_area, target_point)

    def _handle_docking_control_by_sam(self):
        """Handle docking control logic by SAM"""
        if (self.target is not None
            and 'hand' in self.target
            and self.status != 'DOCKED'
            and self.hand_detection is not None
            and len(self.hand_detection.center_pixel) > 0
            and len(self.hand_detection.box_id) > 0):

            target_id = int(self.target.split('-')[-2])
            target_indices = (self.hand_detection.box_id == target_id).nonzero()[0]

            if len(target_indices) > 0:
                target_idx = target_indices[0]
                target_point = self.hand_detection.center_pixel[target_idx]
                seg = self.sam.predict(self.hand_rgb, points=[target_point], verbose=False)[0]
                mask = seg.masks.data.cpu().numpy().reshape(self.hand_rgb_intrinsics['height'], self.hand_rgb_intrinsics['width'])
                colored_mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2BGR)
                self.colored_mask = (colored_mask * np.array((0, 0, 255))).astype('uint8')
                mask_area = np.sum(mask)

                # Update visual servoing target
                center_px = (self.hand_rgb_intrinsics['width'] // 2, self.hand_rgb_intrinsics['height'] // 2)
                box = np.array([center_px[0] - self.tolerance, center_px[1] - self.tolerance,
                              center_px[0] + self.tolerance, center_px[1] + self.tolerance]).astype(np.int32)

                self.perform_docking_by_box_sam(box, mask_area, target_point)

    def _handle_inserting_control(self):
        """Handle inserting control logic"""
        if self.armstrong_plan_status not in [ARMstrongPlanStatus.IDLE, ARMstrongPlanStatus.COMPLETED, ARMstrongPlanStatus.FAILED]:
            return

        if self.insert_status.is_idle():
            self.insert_status.status = 'APPROACH'
        elif self.insert_status.is_completed():
            self.insert_status.status = 'IDLE'
            self.status = 'COMPLETED'
            return

        wait_after_complete = 0.0
        if self.insert_status.status == 'INSERT':
            trigger = 600
            vel_scale = 0.005
        elif self.insert_status.status == 'FASTEN':
            trigger = 999
            vel_scale = 0.001
            wait_after_complete = 3.0
        else:
            trigger = 512
            vel_scale = 0.015

        self.get_logger().info(f"Inserting status: {self.insert_status.status}")
        self.perform_inserting(self.insert_status.get_current_params(), trigger, vel_scale=vel_scale, wait_after_complete=wait_after_complete)
        self.insert_status.status = self.insert_status.get_next_status()

    def _setup_projected_bolt_box(self):
        """Setup projected bolt box for hand camera"""
        if self.projected_bolt_box_area is None:
            self.bolt_size = 0.036
            self.target_z = 0.145

            self.bolt_box = np.array([
                [-self.bolt_size/2, -self.bolt_size/2, self.target_z],
                [self.bolt_size/2, self.bolt_size/2, self.target_z],
            ])
            self.projected_bolt_box = cv2.projectPoints(
                self.bolt_box,
                np.zeros(3),
                np.zeros(3),
                self.hand_rgb_intrinsics['mtx'],
                self.hand_rgb_intrinsics['dist']
            )[0]
            self.projected_bolt_box = self.projected_bolt_box.reshape(-1, 2).astype(np.int32)
            self.projected_bolt_box_area = np.multiply(*(self.projected_bolt_box[0] - self.projected_bolt_box[1]))

            self.bolt_radius = 0.021
            bolt_circle = np.array([
                [0, 0, self.target_z],
                [self.bolt_radius, 0, self.target_z],
            ])
            self.projected_bolt_circle = cv2.projectPoints(
                bolt_circle,
                np.zeros(3), np.zeros(3), self.hand_rgb_intrinsics['mtx'], self.hand_rgb_intrinsics['dist'])[0]
            self.projected_bolt_circle = self.projected_bolt_circle.reshape(-1, 2).astype(np.int32)
            self.projected_bolt_circle_area = np.sum(np.square((self.projected_bolt_circle[0] - self.projected_bolt_circle[1]))) * np.pi

    def _draw_target_boxes(self, rgb):
        """Draw target boxes on hand camera view"""
        center_px = (self.hand_rgb_intrinsics['width'] // 2, self.hand_rgb_intrinsics['height'] // 2)
        box = np.array([center_px[0] - self.tolerance, center_px[1] - self.tolerance,
                       center_px[0] + self.tolerance, center_px[1] + self.tolerance]).astype(np.int32)
        rgb = cv2.rectangle(rgb, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        rgb = cv2.rectangle(rgb, self.projected_bolt_box[0], self.projected_bolt_box[1], (255, 255, 0), 2)

    def _draw_status_info(self, rgb):
        """Draw status information on image"""
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
        y_offset += line_height
        rgb = cv2.putText(rgb, f'Status: {self.status}', (10, y_offset),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    def _handle_keyboard_input(self):
        """Handle keyboard input for head camera view"""
        k = cv2.waitKey(1) % 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
        elif k == ord('f'):
            self.toggle_fix_pose()
        elif k == ord('r'):
            self.armstrong_plan_status = ARMstrongPlanStatus.IDLE
        elif k == ord('a'):
            self._set_offset_by_imu()
        elif k == ord('i'):
            self.status = 'INSERTING'
            self.insert_pose = self.get_transform('origin', 'r_link6')

    def _set_offset_by_imu(self):
        offset = self.get_param()
        p, y, r  = self.imu_ori.as_euler('xyz', degrees=True)  # as imu's axis differs with robot (X: Right, Y: Down, Z: Forward)
        p += 90
        self.get_logger().info(f"Applying offset: {offset} + {p}")
        offset += p
        self.set_param(offset)
        self.status = 'TUNED'

    def set_target(self, box_id, group_id, prefix: Literal['head', 'hand'] = 'head') -> str:
        """Set target bolt for tracking"""
        self.target = self.make_bolt_name(box_id, group_id, prefix)
        self.get_logger().info(f"Target set to: {self.target}")
        return self.target

    def perform_docking_by_box_sam(self, tol_box, target_bbox_area, target_point):
        """Perform docking operation using box-based approach by SAM"""
        assert self.docking_pose is not None
        if self.armstrong_plan_status not in [ARMstrongPlanStatus.IDLE, ARMstrongPlanStatus.COMPLETED, ARMstrongPlanStatus.FAILED]:
            return

        t_x, t_y = target_point
        is_in_box = tol_box[0] <= t_x <= tol_box[2] and tol_box[1] <= t_y <= tol_box[3]
        is_far_from_box = target_bbox_area < self.projected_bolt_circle_area
        box_ratio = np.clip(target_bbox_area / self.projected_bolt_circle_area, 0, 1)
        cam_z = self.get_abstract_depth_from_box(self.bolt_radius**2 * np.pi, target_bbox_area, self.hand_rgb_intrinsics['fx'], self.hand_rgb_intrinsics['fy'])
        cam_x, cam_y, cam_z = self.pixel_to_camera_coords(t_x, t_y, cam_z, self.hand_rgb_intrinsics['fx'], self.hand_rgb_intrinsics['fy'], self.hand_rgb_intrinsics['cx'], self.hand_rgb_intrinsics['cy'])
        x_move_step = (cam_z - self.target_z)
        y_move_step = -cam_x
        z_move_step = -cam_y

        self.get_logger().info(f"Cam z: {cam_z}, cam x: {cam_x}, cam y: {cam_y}")

        tgt_tf = np.zeros(3)

        self.get_logger().info(f"Move step: {x_move_step}, {y_move_step}, {z_move_step}")
        if is_in_box and not is_far_from_box:
            self.insert_pose = self.get_transform('origin', 'r_link6')
            self.status = 'DOCKED'
            self.get_clock().sleep_for(Duration(seconds=0.5))
            return
        elif is_in_box and is_far_from_box:
            tgt_tf[0] = np.clip(x_move_step, 0.001, np.inf)
        else:
            tgt_tf[1] = y_move_step
            tgt_tf[2] = z_move_step
        self.docking_pose = self.apply_rotation_to_transform(self.docking_pose, tgt_tf)

        msg = self.build_armstrong_plan_request(
            group_name='right_arm',
            link_name='r_link6',
            planning_pipeline='pilz_industrial_motion_planner',
            planner_id='LIN',
            target_pose=self.docking_pose,
            wait_after_complete=1.0,
            vel_scale=0.03,
        )
        self.armstrong_plan_status = ARMstrongPlanStatus.PLANNING
        self.armstrong_plan_pub.publish(msg)


    def get_abstract_depth_from_box(self, area_real, area_px, fx, fy):
        """Get abstract depth from box"""
        z = ((fx * fy * area_real) / area_px) ** (1/2)
        return z

    def pixel_to_camera_coords(self, u, v, Z, fx, fy, cx, cy):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z])

    def apply_rotation_to_transform(self, source_pose: Pose | Transform, target_tls: np.ndarray) -> Pose:
        """Apply rotation to transform"""
        if isinstance(source_pose, Transform):
            source_pos = ros_pos_to_np_pos(source_pose.translation)
            source_rot = ros_quat_to_rotation(source_pose.rotation)
        else:
            source_pos = ros_pos_to_np_pos(source_pose.position)
            source_rot = ros_quat_to_rotation(source_pose.orientation)
        target_tls = source_rot.apply(target_tls) + source_pos
        target_rot = source_rot
        ret_pose = Pose()
        ret_pose.position = list_to_ros_point(target_tls)
        ret_pose.orientation = rot_to_ros_quat(target_rot)
        return ret_pose

    def perform_inserting(self, target: List, target_trigger: int = 512, vel_scale: float = 0.005, wait_after_complete: float = 0.5) -> None:
        """Perform inserting operation"""
        assert self.insert_pose is not None
        if len(target) == 3:
            self.insert_pose = self.apply_rotation_to_transform(self.insert_pose, target)
            msg = self.build_armstrong_plan_request(
                group_name='right_arm',
                link_name='r_link6',
                planner_id='PTP',
                target_pose=self.insert_pose,
                wait_after_complete=wait_after_complete,
                vel_scale=vel_scale,
                trigger=target_trigger,
            )
        elif len(target) == 6:
            joint_state = JointState()
            joint_state.position = target
            joint_state.name = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6']
            msg = self.build_armstrong_plan_request(
                group_name='right_arm',
                link_name='r_link6',
                planner_id='PTP',
                target_joint=joint_state,
                wait_after_complete=wait_after_complete,
                vel_scale=0.1,
                trigger=target_trigger,
            )

        self.armstrong_plan_status = ARMstrongPlanStatus.PLANNING
        self.armstrong_plan_pub.publish(msg)


    def perform_transform(self, source_pose: Tuple[np.ndarray, Rotation], target_transform: Tuple[np.ndarray, Rotation]):
        """Perform transform between source and target poses"""
        source_pos = source_pose[0]
        source_rot = source_pose[1]

        target_tls = target_transform[0]
        target_rot = target_transform[1]

        return_pos = source_rot.apply(target_tls) + source_pos
        return_rot = source_rot * target_rot

        return (return_pos, return_rot)

    def publish_transform(self, target: Tuple[str, np.ndarray, Rotation], source_frame: str) -> None:
        """Publish transform for a target"""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = source_frame
        tf.child_frame_id = source_frame[:4] + '-' + target[0]
        tf.transform.translation = list_to_ros_vec3(target[1])
        tf.transform.rotation = rot_to_ros_quat(target[2])

        self.tf_broadcaster.sendTransform(tf)

    def follow_target(self, target_pose: Pose) -> None:
        """Follow target pose"""
        msg = self.build_armstrong_plan_request(
            group_name='right_arm',
            link_name='r_link6',
            target_pose=target_pose,
            wait_after_complete=0.1,
            vel_scale=0.2,
        )
        if self.armstrong_plan_status in [ARMstrongPlanStatus.IDLE, ARMstrongPlanStatus.COMPLETED, ARMstrongPlanStatus.FAILED]:
            self.armstrong_plan_status = ARMstrongPlanStatus.PLANNING
            self.armstrong_plan_pub.publish(msg)

    def make_bolt_name(self, box_id, group_id, prefix: Literal['head', 'hand'] = 'head'):
        """Create bolt name from components"""
        return '-'.join(map(str, [prefix, group_id, box_id, 'bolt'] if group_id is not None else ['unknown', box_id, 'bolt']))

    def get_target_pose(self, bolt_to_follow: str, tgt_offset = None, link_offset = None):
        """Get target pose for a bolt"""
        tf = self.get_transform('origin', bolt_to_follow)
        if tf is None:
            return

        pos = ros_pos_to_np_pos(tf.position)
        rot = ros_quat_to_rotation(tf.orientation)
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

    def get_transform(self, origin, target) -> Pose | None:
        """Get transform between origin and target"""
        now = self.get_clock().now()
        if not self.tf_buffer.can_transform(origin, target, Time()):
            return

        ts = self.tf_buffer.lookup_transform(origin, target, Time())
        return transform_to_pose(ts.transform)

    def toggle_fix_pose(self):
        """Toggle fix pose mode"""
        self.is_head_update_detection = not self.is_head_update_detection

    def toggle_camera_location(self, location: Optional[Literal['head', 'hand']] = None):
        """Toggle camera location"""
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
        """Build ARMstrong plan request message"""
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



    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self.get_logger().info(f"Status Changed: {self._status} -> {value}")
        self._status = value

def main():
    """Main function"""
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
