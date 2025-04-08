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

    glossy_image: None
    diffuse_image: None
    position_image: None
    normal_image: None
    roughness_image: None
    metalness_image: None
    base_color_image: None
    brdf_image: None
    specular_image: None
