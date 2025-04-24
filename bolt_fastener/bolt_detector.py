from typing import List
from numpy.typing import NDArray, ArrayLike
import torch

from scipy.spatial.transform import Rotation

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2
import numpy as np
from armstrong_py.filters import ExponentialMovingAverage


class BoltDetector:
    def __init__(self, model_path='./best.pt'):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.cam_info = {}
        self.alpha = 0.1  # (1/15)/((1/15)+0.1)
        self.box_smoothing = {}  # Dictionary to store moving averages for each box ID

    def detect(self, img: NDArray) -> Boxes | None:
        res = self.model.predict(img)
        if res is None:
            return None
        if res[0].boxes is None:
            return None
        
        return res[0].boxes
    
    def track(self, img: NDArray) -> 'DetectionResult | None':
        res = self.model.track(img, persist=True, verbose=False)
        if res is None:
            return None
        if res[0].boxes is None:
            return None

        boxes = res[0].boxes.cpu()
        if boxes.id is None or boxes.conf is None or boxes.xyxy is None:
            return None

        # Apply smoothing to bounding boxes
        for i, box_id in enumerate(boxes.id):
            if box_id not in self.box_smoothing:
                # Initialize moving average for new box
                self.box_smoothing[box_id] = ExponentialMovingAverage(self.alpha)
            
            # Update moving average and get smoothed coordinates
            with torch.inference_mode(True):
                smoothed_coords = self.box_smoothing[box_id].update(boxes.xyxy[i])
                boxes.xyxy[i] = smoothed_coords
        
        return DetectionResult(boxes)
    
    def draw(self, img: NDArray, detection: 'DetectionResult', bolts: List[tuple[str, NDArray, Rotation]]) -> NDArray:
        to_zipped = [detection.xyxy,
            detection.box_id,
            detection.confidence,
            ]
        if detection.group_id is not None:
            to_zipped.append(detection.group_id)
        
        for i, x in enumerate(zip(*to_zipped)):
            loc = x[0]
            idx = x[1]
            conf = x[2]
            group_id = x[3] if len(x) > 3 else None
            s = f'{group_id}-{idx}: {conf:.2f}'
            img = cv2.putText(img, s, loc[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img = cv2.rectangle(img, loc[:2], loc[2:], (0, 255, 0), 2)
            img = cv2.circle(img, np.mean((loc[:2], loc[2:]), axis=0).astype(np.int32), 5, (0, 0, 255), -1)
            if self.cam_info != {} and len(bolts) > 0:
                name, tvec, rot = bolts[i]
                # tvec = [tvec[1], tvec[0], tvec[2]]
                rvec = rot.as_rotvec()
                img = cv2.drawFrameAxes(img, self.cam_info['mtx'], self.cam_info['dist'], rvec, tvec, 0.05, 1)

        return img


class DetectionResult:
    def __init__(self, detection: Boxes | List | None, ):
        if isinstance(detection, Boxes):
            detection = detection.cpu()
            xyxy = detection.xyxy.numpy().astype(int)
            self.xyxy: NDArray[np.int_] = xyxy
            self.box_id: NDArray[np.int_] = detection.id.numpy().astype(int)
            self.confidence: NDArray[np.float_] = detection.conf.numpy().astype(float)
            self.center_pixel: NDArray[np.int_] = np.stack(((xyxy[:, 0] + xyxy[:, 2]) // 2, ((xyxy[:, 1] + xyxy[:, 3]) // 2)), axis=-1)
            self.center_point: NDArray[np.float_] | None = None
            self.group_id: NDArray[np.int_] | None = None

        if isinstance(detection, List):
            self.xyxy = detection[0]
            self.box_id = detection[1]
            self.confidence = detection[2]
            self._center_pixel = detection[3]
            self.center_point = detection[4] if len(detection) > 4 else None
            self.group_id = detection[5] if len(detection) > 5 else None
    
    def __len__(self):
        return len(self.xyxy)
    
    def __getitem__(self, idx_: int | str):
        if isinstance(idx_, str):
            s = idx_.split('-')
            idx_ = int(s[-2])

        idx = (self.box_id == idx_).nonzero()[0][0]
        
        it = DetectionResult([
                    self.xyxy[idx],
                    self.box_id[idx],
                    self.confidence[idx],
                    self.center_pixel[idx],
                    self.center_point[idx] if self.center_point is not None else None,
                    self.group_id[idx] if self.group_id is not None else None
                ])
        return it

    def filter_with_mask(self, mask: np.ndarray, inplace=True):
        if inplace:
            self.xyxy = self.xyxy[mask]
            self.box_id = self.box_id[mask]
            self.confidence = self.confidence[mask]
            self.center_pixel = self.center_pixel[mask]
            if self.center_point is not None:
                self.center_point = self.center_point[mask]
            if self.group_id is not None:
                self.group_id = self.group_id[mask]
            return mask[mask]
        else:
            ret = []
            ret.append(self.xyxy[mask])
            ret.append(self.box_id[mask])
            ret.append(self.confidence[mask])
            ret.append(self.center_pixel[mask])
            if self.center_point is not None:
                ret.append(self.center_point[mask])
            if self.group_id is not None:
                ret.append(self.group_id[mask])
            return DetectionResult(ret), mask[mask]
        