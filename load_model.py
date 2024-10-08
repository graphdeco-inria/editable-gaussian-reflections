import torch 
from scene import GaussianModel
from scene import cameraList_from_camInfos
from scene.dataset_readers import readNerfSyntheticInfo
import copy
from scene.cameras import *
import numpy as np 

class args:
    resolution = 800
    data_device = "cuda"
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


T = np.array([-2.53716236e-11, -6.97627801e-09,  1.92176780e+00]) 
R = np.array([[ 9.99989839e-01,  2.29605934e-03, -3.87952831e-03],
       [ 4.50806260e-03, -5.09317711e-01,  8.60566759e-01],
       [-1.34420959e-10, -8.60575503e-01, -5.09322887e-01]])
camera_colmap = Camera(0, R, T, 0.6911121570925639, 0.4710906705955732, torch.zeros(3, 1024, 1536), None, None, None, None, None, None, None)


[
                [
                    0.9999898672103882,
                    -0.002296059625223279,
                    0.00387952895835042,
                    
                ],
                [
                    0.004508062731474638,
                    0.5093177556991577,
                    -0.8605669140815735,
                    
                ],
                [
                    -2.3283064365386963e-10,
                    0.8605756163597107,
                    0.509323000907898,
                    
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]

T = np.array([0.007455553859472275, -1.6538097858428955, 0.9788005352020264]) 
R = np.array([[ 0.9999898672103882,
                    -0.002296059625223279,
                    0.00387952895835042],
       [ 0.004508062731474638,
                    0.5093177556991577,
                    -0.8605669140815735],
       [ -2.3283064365386963e-10,
                    0.8605756163597107,
                    0.509323000907898]])
camera = Camera(0, R, T, 0.6911121570925639, 0.4710906705955732, torch.zeros(3, 1024, 1536), None, None, None, None, None, None, None)

# scene_info = readNerfSyntheticInfo("./data/sphere_smooth", False, False)
# train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args, max_images=1)
# camera = train_cameras[0]
# camera = Camera(0, camera.R, camera.T, 0.691111147403717, 0.691111147403717, torch.zeros(3, 800, 800), None, None, None)