from dataclasses import dataclass

import numpy as np


@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

    specular_image: np.array
    diffuse_image: np.array
    depth_image: np.array
    normal_image: np.array
    roughness_image: np.array
    f0_image: np.array
