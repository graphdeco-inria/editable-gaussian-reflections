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

def imread(image_path, render_pass_name):
    path = image_path.replace("/images/", "/render/").replace("/colmap/", "/renders/").replace("/render_", f"/{render_pass_name}_").replace("/render/", f"/{render_pass_name}/")
    path = path.replace(".png", ".exr")
    assert os.path.exists(path), f"{render_pass_name} render pass not found at {path}"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image
    
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

        if "LOAD_FROM_IMAGE_FILES" not in os.environ:
            cache_path = image_path.replace("/renders/", "/cache/").replace("render/render_", "").replace(".png", ".pt")
            image_tensor = torch.load(cache_path)
            image, diffuse_image, glossy_image, normal_image, position_image, roughness_image, specular_image, metalness_image, base_color_image, brdf_image = torch.unbind(image_tensor, dim=0)
            height, width = image.shape[0], image.shape[1]
        else:
            image = imread(image_path, "render") 
            diffuse_image = imread(image_path, "diffuse") 
            glossy_image = imread(image_path, "glossy") 
            normal_image = imread(image_path, "normal") 
            position_image = imread(image_path, "position")
            roughness_image = imread(image_path, "roughness")
            specular_image = imread(image_path, "specular")
            metalness_image = imread(image_path, "metalness")
            base_color_image = imread(image_path, "base_color")
            brdf_image = imread(image_path, "glossy_brdf")
            height, width = image.shape[0], image.shape[1]
        diffuse_image = diffuse_image * model_params.exposure
        glossy_image = glossy_image * model_params.exposure

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, diffuse_image=diffuse_image, glossy_image=glossy_image, position_image=position_image, normal_image=normal_image, roughness_image=roughness_image, metalness_image=metalness_image, base_color_image=base_color_image, brdf_image=brdf_image, specular_image=specular_image)

        cam_infos.append(cam_info)
    
    futures = []
    with ThreadPoolExecutor() as executor: 
        for idx, key in tqdm(keys):
            futures.append(executor.submit(readFrame, idx, key))

    for future in concurrent.futures.as_completed(futures):
        future.result()

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    
    sys.stdout.write('\n')
    return cam_infos

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
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    rgb = rgb / 255.0
    
    # We create random points inside the bounds of the synthetic Blender scenes
    num_rand_pts = model_params.num_farfield_init_points
    print(f"Generating random point cloud ({num_rand_pts})...")
    rand_xyz = (np.random.random((num_rand_pts, 3)) * 2.6 - 1.3) * model_params.glossy_bbox_size_mult #!!! todo only if glossy 
    rand_rgb = np.random.random((num_rand_pts, 3))
    points = np.concatenate([xyz, rand_xyz], axis=0)
    colors = np.concatenate([rgb, rand_rgb], axis=0)

    pcd = BasicPointCloud(
        points=points,
        colors=colors, 
        normals=np.zeros_like(points),
    )
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

            assert model_params.linear_space
            
            if "LOAD_FROM_IMAGE_FILES" not in os.environ:
                cache_path = image_path.replace("/renders/", "/cache/").replace("render/render_", "").replace(".png", ".pt")
                image_tensor = torch.load(cache_path)
                image, diffuse_image, glossy_image, normal_image, position_image, roughness_image, metalness_image, base_color_image, brdf_image, specular_image = torch.unbind(image_tensor, dim=0)
                height, width = image.shape[0], image.shape[1]
            else:
                image = imread(image_path, "render") 
                diffuse_image = imread(image_path, "diffuse") 
                glossy_image = imread(image_path, "glossy") 
                normal_image = imread(image_path, "normal") 
                position_image = imread(image_path, "position")
                roughness_image = imread(image_path, "roughness")
                specular_image = imread(image_path, "specular")
                metalness_image = imread(image_path, "metalness")
                base_color_image = imread(image_path, "base_color")
                brdf_image = imread(image_path, "glossy_brdf")
                height, width = image.shape[0], image.shape[1]
            diffuse_image = diffuse_image * model_params.exposure
            glossy_image = glossy_image * model_params.exposure

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, diffuse_image=diffuse_image, glossy_image=glossy_image, position_image=position_image, normal_image=normal_image, roughness_image=roughness_image, metalness_image=metalness_image, base_color_image=base_color_image, brdf_image=brdf_image, specular_image=specular_image))


        with ThreadPoolExecutor() as executor: 
            futures = []
            for idx, frame in tqdm(enumerate(frames)):
                futures.append(executor.submit(readFrame, idx, frame))
            
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cam_infos) > 0

        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
            
    return cam_infos

def readNerfSyntheticInfo(model_params, path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(model_params, path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(model_params, path, "transforms_test.json", white_background, extension)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    rgb = rgb / 255.0
    
    # We create random points inside the bounds of the synthetic Blender scenes
    num_rand_pts = model_params.num_farfield_init_points
    print(f"Generating random point cloud ({num_rand_pts})...")
    rand_xyz = (np.random.random((num_rand_pts, 3)) * 2.6 - 1.3) * model_params.glossy_bbox_size_mult #!!! todo only if glossy 
    rand_rgb = np.random.random((num_rand_pts, 3))
    points = np.concatenate([xyz, rand_xyz], axis=0)
    colors = np.concatenate([rgb, rand_rgb], axis=0)

    pcd = BasicPointCloud(
        points=points,
        colors=colors, 
        normals=np.zeros_like(points),
    )
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