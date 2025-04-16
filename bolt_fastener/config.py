from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, cast
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_service import SetParametersResult

@dataclass
class BoltFastenerConfig:
    """Configuration parameters for the BoltFastener system"""
    # Camera settings
    use_tcp_tunnel: bool = True
    camera_namespace: str = '/camera/camera'
    
    # Detection settings
    confidence_threshold: float = 0.5
    detection_update_rate: float = 30.0  # Hz
    
    # Point cloud processing
    roi_offset: float = 0.1
    radius: float = 0.04
    eps: float = 0.4
    
    # Robot control
    vel_scale: float = 0.2
    wait_after_complete: float = 0.1
    target_offset: List[float] = [-0.30, 0, 0.02]
    
    # Visualization
    show_fps: bool = True
    show_detection_status: bool = True
    show_target_status: bool = True
    show_planning_status: bool = True

class ConfigManager:
    def __init__(self, node: Node):
        self.node = node
        self.config = BoltFastenerConfig()
        
        # Declare parameters
        self._declare_parameters()
        
        # Load initial values
        self._load_parameters()
        
        # Set up parameter callback
        self.node.add_on_set_parameters_callback(self._parameter_callback)

    def _declare_parameters(self):
        """Declare all ROS2 parameters"""
        # Camera settings
        self.node.declare_parameter('use_tcp_tunnel', self.config.use_tcp_tunnel)
        self.node.declare_parameter('camera_namespace', self.config.camera_namespace)
        
        # Detection settings
        self.node.declare_parameter('confidence_threshold', self.config.confidence_threshold)
        self.node.declare_parameter('detection_update_rate', self.config.detection_update_rate)
        
        # Point cloud processing
        self.node.declare_parameter('roi_offset', self.config.roi_offset)
        self.node.declare_parameter('radius', self.config.radius)
        self.node.declare_parameter('eps', self.config.eps)
        
        # Robot control
        self.node.declare_parameter('vel_scale', self.config.vel_scale)
        self.node.declare_parameter('wait_after_complete', self.config.wait_after_complete)
        self.node.declare_parameter('target_offset', self.config.target_offset)
        
        # Visualization
        self.node.declare_parameter('show_fps', self.config.show_fps)
        self.node.declare_parameter('show_detection_status', self.config.show_detection_status)
        self.node.declare_parameter('show_target_status', self.config.show_target_status)
        self.node.declare_parameter('show_planning_status', self.config.show_planning_status)

    def _load_parameters(self):
        """Load parameter values from ROS2 parameter server"""
        # Camera settings
        self.config.use_tcp_tunnel = bool(self.node.get_parameter('use_tcp_tunnel').get_parameter_value().bool_value)
        self.config.camera_namespace = str(self.node.get_parameter('camera_namespace').get_parameter_value().string_value)
        
        # Detection settings
        self.config.confidence_threshold = float(self.node.get_parameter('confidence_threshold').get_parameter_value().double_value)
        self.config.detection_update_rate = float(self.node.get_parameter('detection_update_rate').get_parameter_value().double_value)
        
        # Point cloud processing
        self.config.roi_offset = float(self.node.get_parameter('roi_offset').get_parameter_value().double_value)
        self.config.radius = float(self.node.get_parameter('radius').get_parameter_value().double_value)
        self.config.eps = float(self.node.get_parameter('eps').get_parameter_value().double_value)
        
        # Robot control
        self.config.vel_scale = float(self.node.get_parameter('vel_scale').get_parameter_value().double_value)
        self.config.wait_after_complete = float(self.node.get_parameter('wait_after_complete').get_parameter_value().double_value)
        self.config.target_offset = cast(List[float], self.node.get_parameter('target_offset').get_parameter_value().double_array_value)
        
        # Visualization
        self.config.show_fps = bool(self.node.get_parameter('show_fps').get_parameter_value().bool_value)
        self.config.show_detection_status = bool(self.node.get_parameter('show_detection_status').get_parameter_value().bool_value)
        self.config.show_target_status = bool(self.node.get_parameter('show_target_status').get_parameter_value().bool_value)
        self.config.show_planning_status = bool(self.node.get_parameter('show_planning_status').get_parameter_value().bool_value)

    def _parameter_callback(self, params: List[Parameter]) -> SetParametersResult:
        """Handle parameter updates"""
        for param in params:
            if param.name == 'confidence_threshold':
                self.config.confidence_threshold = float(param.get_parameter_value().double_value)
            elif param.name == 'detection_update_rate':
                self.config.detection_update_rate = float(param.get_parameter_value().double_value)
            elif param.name == 'vel_scale':
                self.config.vel_scale = float(param.get_parameter_value().double_value)
            # Add more parameter handlers as needed
            
        return SetParametersResult(successful=True)

    def get_config(self) -> BoltFastenerConfig:
        """Get the current configuration"""
        return self.config 