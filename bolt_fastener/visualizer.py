from typing import Optional, Tuple, List
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .bolt_detector import BoltDetector, DetectionResult

class Visualizer:
    def __init__(self):
        self.fps = 0
        self.frame_time = []
        self.prev_time = None
        self.alpha = 0.1  # Smoothing factor for rolling average
        
        # Initialize windows and trackbars
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

    def update_fps(self, current_time):
        """Update FPS calculation"""
        if self.prev_time is None:
            self.prev_time = current_time
            return
        
        self.frame_time.append((current_time - self.prev_time).nanoseconds)
        if len(self.frame_time) > 100:
            self.frame_time.pop(0)
        self.fps = 1 / np.mean(self.frame_time) * 1e9
        self.prev_time = current_time

    def process_depth_image(self, depth: np.ndarray) -> np.ndarray:
        """Process depth image for visualization"""
        try:
            depth = depth.copy()
            min_ = cv2.getTrackbarPos('min', 'depth') / 1000
            max_ = cv2.getTrackbarPos('max', 'depth') / 1000
            
            depth = np.clip(depth, min_, max_)
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            cv2.normalize(depth, depth_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        except Exception as e:
            raise RuntimeError(f"Error processing depth image: {str(e)}")

    def draw_status(self, 
                   img: np.ndarray, 
                   is_update_detection: bool,
                   target: Optional[str],
                   planning_status: str,
                   status_color: Tuple[int, int, int]) -> np.ndarray:
        """Draw status information on the image"""
        y_offset = 30
        line_height = 30
        
        # Performance metrics
        img = cv2.putText(img, f'FPS: {self.fps:.2f}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Detection status
        img = cv2.putText(img, f'Detection: {"ON" if is_update_detection else "OFF"}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if is_update_detection else (0, 0, 255), 2)
        y_offset += line_height
        
        # Target status
        img = cv2.putText(img, f'Target: {target}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += line_height
        
        # Planning status
        img = cv2.putText(img, f'Planning: {planning_status}', (10, y_offset), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return img

    def get_trackbar_values(self) -> Tuple[float, float, float]:
        """Get current trackbar values"""
        contrast = cv2.getTrackbarPos('Contrast', 'rgb')
        brightness = cv2.getTrackbarPos("Brightness", 'rgb')
        confidence = cv2.getTrackbarPos("Confidence", 'rgb') / 100.0
        return contrast, brightness, confidence

    def process_rgb_image(self, img: np.ndarray) -> np.ndarray:
        """Process RGB image with current trackbar settings"""
        contrast, brightness, _ = self.get_trackbar_values()
        return cv2.convertScaleAbs(img, alpha=(contrast-50.0)/50+1, beta=brightness-50)

    def show_images(self, rgb: np.ndarray, depth: np.ndarray) -> int:
        """Display images and handle key events"""
        cv2.imshow('rgb', rgb)
        cv2.imshow('depth', depth)
        return cv2.waitKey(1) % 0xff

    def cleanup(self):
        """Cleanup visualization resources"""
        cv2.destroyAllWindows() 