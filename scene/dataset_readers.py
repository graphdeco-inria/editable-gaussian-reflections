#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from dataclasses import dataclass
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
from tqdm import tqdm 
import concurrent 
import torch

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

@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(model_params, cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    keys = sorted(enumerate(cam_extrinsics), key=lambda x: x[0])[:model_params.max_images]

    def readFrame(idx, key):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path.replace("/colmap/", "/renders/").replace("/images/", "/render/"))

        def imread(image_path, image_name):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert image is not None, f"{image_name} image not found at {image_path}"
            return image
        
        diffuse_image_path = image_path.replace("/render_", "/diffuse_").replace("/colmap/", "/renders/").replace("/images/", "/diffuse/").replace(".png", ".exr")
        diffuse_image = imread(diffuse_image_path, "Diffuse") * model_params.exposure

        glossy_image_path = image_path.replace("/render_", "/glossy_").replace("/colmap/", "/renders/").replace("/images/", "/glossy/").replace(".png", ".exr")
        glossy_image = imread(glossy_image_path, "Glossy") * model_params.exposure
        
        normal_image_path = image_path.replace("/render_", "/normal_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/normal/")
        normal_image = imread(normal_image_path, "Normal")

        position_image_path = image_path.replace("/render_", "/position_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/position/")
        position_image = imread(position_image_path, "Position")

        roughness_image_path = image_path.replace("/render_", "/roughness_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/roughness/")
        roughness_image = imread(roughness_image_path, "Roughness")

        specular_image_path = image_path.replace("/render_", "/specular_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/specular/")
        specular_image = imread(specular_image_path, "Specular")

        metalness_path = image_path.replace("/render_", "/metalness_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/metalness/")
        metalness_image = imread(metalness_path, "Metalness")

        base_color_path = image_path.replace("/render_", "/base_color_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/base_color/")
        base_color_image = imread(base_color_path, "BaseColor")

        brdf_path = image_path.replace("/render_", "/glossy_brdf_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/images/", "/glossy_brdf/")
        brdf_image = imread(brdf_path, "BRDF")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, diffuse_image=diffuse_image, glossy_image=glossy_image, position_image=position_image, normal_image=normal_image, roughness_image=roughness_image, metalness_image=metalness_image, base_color_image=base_color_image, brdf_image=brdf_image, specular_image=specular_image)

        cam_infos.append(cam_info)

    cache_path = os.path.join(model_params.source_path, "cam_infos.pth")
    if "DBGCACHE" in os.environ and os.path.exists(cache_path):
        print("Loading", cache_path)
        
        cam_infos = torch.load(cache_path)
        if len(cam_infos) == model_params.max_images:
            return cam_infos
        else:
            print("Cache invalid")
    
    futures = []
    with ThreadPoolExecutor() as executor: 
        for idx, key in tqdm(keys):
            futures.append(executor.submit(readFrame, idx, key))

    for future in concurrent.futures.as_completed(futures):
        future.result()

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    if "DBGCACHE" in os.environ:
        torch.save(cam_infos, cache_path)
    
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(model_params, path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(model_params, cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    assert len(cam_infos_unsorted) > 0, "No cameras found in the scene!"
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

from concurrent.futures import ThreadPoolExecutor


def readCamerasFromTransforms(model_params, path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = sorted(contents["frames"], key=lambda frame: frame["file_path"])[:model_params.max_images]

        cache = None
        index_mapping = {
            "Render": slice(0, 3),
            "Diffuse": slice(3, 6),
            "Glossy": slice(6, 9),
            "Normal": slice(9, 12),
            "Position": slice(12, 15),
            "Roughness": slice(15, 18),
            "Metalness": slice(18, 21),
            "BaseColor": slice(21, 24),
            "BRDF": slice(24, 27),
            "Specular": slice(27, 30)
        }
        
        def imread(image_path, image_name):
            if cache is None:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                assert image is not None, f"{image_name} image not found at {image_path}"
                return image
            else:
                return cache[idx, index_mapping[image_name]]

        def readFrame(idx, frame):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            assert model_params.linear_space
            
            diffuse_image_path = image_path.replace("/render_", "/diffuse_").replace("/colmap/", "/renders/").replace("/render/", "/diffuse/").replace(".png", ".exr")
            diffuse_image = imread(diffuse_image_path, "Diffuse") * model_params.exposure

            glossy_image_path = image_path.replace("/render_", "/glossy_").replace("/colmap/", "/renders/").replace("/render/", "/glossy/").replace(".png", ".exr")
            glossy_image = imread(glossy_image_path, "Glossy") * model_params.exposure
            
            normal_image_path = image_path.replace("/render_", "/normal_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/normal/")
            normal_image = imread(normal_image_path, "Normal")

            position_image_path = image_path.replace("/render_", "/position_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/position/")
            position_image = imread(position_image_path, "Position")

            roughness_image_path = image_path.replace("/render_", "/roughness_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/roughness/")
            roughness_image = imread(roughness_image_path, "Roughness")

            specular_image_path = image_path.replace("/render_", "/specular_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/specular/")
            specular_image = imread(specular_image_path, "Specular")

            metalness_path = image_path.replace("/render_", "/metalness_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/metalness/")
            metalness_image = imread(metalness_path, "Metalness")

            base_color_path = image_path.replace("/render_", "/base_color_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/base_color/")
            base_color_image = imread(base_color_path, "BaseColor")

            brdf_path = image_path.replace("/render_", "/glossy_brdf_").replace(".png", ".exr").replace("/colmap/", "/renders/").replace("/render/", "/glossy_brdf/")
            brdf_image = imread(brdf_path, "BRDF")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], diffuse_image=diffuse_image, glossy_image=glossy_image, position_image=position_image, normal_image=normal_image, roughness_image=roughness_image, metalness_image=metalness_image, base_color_image=base_color_image, brdf_image=brdf_image, specular_image=specular_image))


        cache_path = os.path.join(model_params.source_path, "cam_infos_" + str(model_params.max_images) + "images_" + str(model_params.resolution) + "px_" + transformsfile.replace(".json", ".pt"))

        if "DBGCACHE" in os.environ and os.path.exists(cache_path):
            print("Loading", cache_path)
            
            cam_infos = torch.load(cache_path)
            if len(cam_infos) == model_params.max_images: # todo just put the # of images and the resolution in the file name
                return cam_infos
            else:
                print("Cache invalid")
                
        with ThreadPoolExecutor() as executor: 
            futures = []
            for idx, frame in tqdm(enumerate(frames)):
                futures.append(executor.submit(readFrame, idx, frame))
            
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cam_infos) > 0

        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
        if "DBGCACHE" in os.environ:
            torch.save(cam_infos, cache_path)

        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
            
    return cam_infos

def readNerfSyntheticInfo(model_params, path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(model_params, path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(model_params, path, "transforms_test.json", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    num_pts = model_params.num_farfield_init_points
    print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (np.random.random((num_pts, 3)) * 2.6 - 1.3) * model_params.glossy_bbox_size_mult #!!! todo only if glossy 
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)   
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}