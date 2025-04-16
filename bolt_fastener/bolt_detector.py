from typing import List
from numpy.typing import NDArray, ArrayLike

from scipy.spatial.transform import Rotation

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2
import numpy as np


class BoltDetector:
    def __init__(self, model_path='./best.pt'):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.cam_info = {}

    def detect(self, img: NDArray) -> Boxes | None:
        res = self.model.predict(img, )
        if res is None:
            return None
        if res[0].boxes is None:
            return None
        
        box = res[0].boxes
        
        return box.cpu()
    
    def track(self, img: NDArray) -> 'DetectionResult | None':
        res = self.model.track(img, persist=True, verbose=False)
        if res is None:
            return None
        if res[0].boxes is None:
            return None

        if (res[0].boxes.id is None 
            or res[0].boxes.conf is None 
            or res[0].boxes.xyxy is None):
            return

        return DetectionResult(res[0].boxes)
    
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
            if self.cam_info != {}:
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
        