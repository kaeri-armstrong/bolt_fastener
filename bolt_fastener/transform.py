from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def perform_transform(source_pose: Tuple[np.ndarray, Rotation], target_transform: Tuple[np.ndarray, Rotation]):
    source_pos = source_pose[0]
    source_rot = source_pose[1]

    target_tls = target_transform[0]
    target_rot = target_transform[1]

    return_pos = source_rot.apply(target_tls) + source_pos
    return_rot = source_rot * target_rot

    return (return_pos, return_rot)