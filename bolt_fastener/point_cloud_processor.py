from typing import List, Tuple, Optional, Literal
from numpy.typing import NDArray

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.linalg import svd
from sklearn.cluster import DBSCAN

from .bolt_detector import DetectionResult

class PointCloudProcessor:
    def __init__(self, depth_intrinsics: Optional[dict]=None):
        if depth_intrinsics is not None:
            self.fx = depth_intrinsics['fx']
            self.fy = depth_intrinsics['fy']
            self.cx = depth_intrinsics['cx']
            self.cy = depth_intrinsics['cy']

            self.height = depth_intrinsics["height"]
            self.width = depth_intrinsics["width"]


    def set_depth_intrinsics(self, depth_intrinsics: dict):
        self.fx = depth_intrinsics['fx']
        self.fy = depth_intrinsics['fy']
        self.cx = depth_intrinsics['cx']
        self.cy = depth_intrinsics['cy']

        self.height = depth_intrinsics["height"]
        self.width = depth_intrinsics["width"]

    def convert_depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        # Convert depth image to point cloud
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        z = depth_image
        y = (v - self.cy) * z / self.fy
        x = (u - self.cx) * z / self.fx

        return np.stack([x, y, z], axis=-1)
    
    def slice_point_cloud_by_roi(self, point_cloud: np.ndarray, roi: tuple) -> np.ndarray:
        x_min, y_min, x_max, y_max = roi
        mask = (point_cloud[:, :, 0] >= x_min) & (point_cloud[:, :, 0] <= x_max) & \
               (point_cloud[:, :, 1] >= y_min) & (point_cloud[:, :, 1] <= y_max)
        return point_cloud[mask]

    
    def get_roi_by_centers(self, centers: NDArray, offset=0.1) -> tuple:
        x_min = min(centers[:, 0]) - offset
        y_min = min(centers[:, 1]) - offset
        x_max = max(centers[:, 0]) + offset
        y_max = max(centers[:, 1]) + offset
        return (x_min, y_min, x_max, y_max)
    
    def get_nearby_points_from_point_cloud_kdtree(self, point_cloud: np.ndarray, centers: List[np.ndarray], radius: float = 0.05) -> List[np.ndarray]:
        p = point_cloud.reshape(-1, 3)
        kdtree = KDTree(p)
        nearby_points_indices = [kdtree.query_ball_point(center, radius) for center in centers]
        ret = [p[i] for i in nearby_points_indices]
        return ret
    
    def get_normal_vector_from_point_cloud_svd(self, points: np.ndarray) -> np.ndarray:
        mean = np.mean(points, axis=0)
        dist = points - mean
        _, _, vh = svd(dist, full_matrices=False)
        normal = vh[-1]  # The last singular vector
        if normal[np.abs(normal).argmax()] < 0:
            normal = -normal
        return normal / np.linalg.norm(normal)

    def get_rotation_from_normal_vector(self, normal_vector: np.ndarray) -> Rotation:
        up = [0, 0, 1]
        r, _ = Rotation.align_vectors(normal_vector, up)
        return r
    
    def cluster_by_center_of_bounding_box(self, points: np.ndarray, eps=0.4) -> np.ndarray:
        group = DBSCAN(eps=eps, min_samples=3).fit_predict(points)
        return np.array(group)
    
    def estimate_bolt_orientation_group(self, bolts: np.ndarray) -> Rotation:
        normal = self.get_normal_vector_from_point_cloud_svd(bolts)
        rot = self.get_rotation_from_normal_vector(normal)
        return rot

    def estimate_bolt_orientation_single(self, point_cloud: np.ndarray) -> Rotation:
        normal = self.get_normal_vector_from_point_cloud_svd(point_cloud)
        rot = self.get_rotation_from_normal_vector(normal)
        return rot            
    
    def estimate_bolt_pose_pipeline(self, 
                                    depth: np.ndarray,
                                    detection: DetectionResult,
                                    conf: Optional[float] = None,
                                    eps: float=0.4,
                                    crop_by_roi: bool=False, 
                                    roi_offset: float=0.1,
                                    radius: float=0.04,
                                    force_single_estimation: bool=False,
                                    ) -> Tuple[List[Tuple[str, NDArray[np.int_], Rotation]], DetectionResult] | None:
        point_cloud = self.convert_depth_to_point_cloud(depth)
        center_pixel = detection.center_pixel
        center_point: NDArray[np.float_] = point_cloud[center_pixel[:, 1], center_pixel[:, 0]]
        detection.center_point = center_point

        if len(detection) == 0:
            return
        group_id = self.cluster_by_center_of_bounding_box(detection.center_point, eps=eps)
        detection.group_id = group_id

        mask = group_id != -1
        mask = detection.filter_with_mask(mask)

        if conf is not None:
            mask = detection.confidence > conf
            mask = detection.filter_with_mask(mask)

        groups, count_groups = np.unique(detection.group_id, return_counts=True)
        bolts = []
        
        for g, c in zip(groups, count_groups):
            if g == -1:
                continue
            
            mask = detection.group_id == g
            d, mask = detection.filter_with_mask(mask, inplace=False)

            if len(d.center_point) == 0:
                continue

            if c >= 3 and not force_single_estimation:
                orientation = self.estimate_bolt_orientation_group(d.center_point)
                for i in range(c):
                    bolt = (f'{d.group_id[i]}-{d.box_id[i]}-bolt', d.center_point[i], orientation)
                    bolts.append(bolt)
            else:
                if crop_by_roi:
                    roi = self.get_roi_by_centers(d.center_point, roi_offset)
                    point_cloud = self.slice_point_cloud_by_roi(point_cloud, roi)

                points = self.get_nearby_points_from_point_cloud_kdtree(point_cloud, d.center_point, radius=radius)
                for i, p in enumerate(points):
                    if len(p) == 0:
                        continue
                    orientation = self.estimate_bolt_orientation_single(p)
                    bolt = (f'{d.group_id[i]}-{d.box_id[i]}-bolt', d.center_point[i], orientation)
                    bolts.append(bolt)
        
        if conf is not None:
            conf_mask = detection.confidence > conf
            detection.filter_with_mask(conf_mask)

        group_mask = detection.group_id != -1
        detection.filter_with_mask(group_mask)

        return bolts, detection
