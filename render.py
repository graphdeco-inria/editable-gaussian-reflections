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

import math
import os
import shutil
from os import makedirs
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional
import json 

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import tyro
from tyro.conf import arg
from tqdm import tqdm
import warnings

from gaussian_tracing.arguments import (
    TyroConfig,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.general_utils import set_seeds
from gaussian_tracing.utils.image_utils import psnr
from gaussian_tracing.utils.tonemapping import tonemap
from gaussian_tracing.utils.system_utils import searchForMaxIteration

@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Optional[int] = None
    spp: int = 128
    split: Literal["train", "test"] = "test"
    denoise: bool = True
    modes: list[Literal["regular", "env_rot_1", "env_move_1", "env_move_2"]] = field(
        default_factory=lambda: ["regular"]
    )
    skip_video: bool = False
    skip_save_frames: bool = False
    znear: float = 0.01

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")

@torch.no_grad()
def render_set(
    cli: RenderCLI,
    cfg: TyroConfig,
    scene,
    split,
    iteration,
    views,
    raytracer,
):

    for mode in cli.modes:
        render_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "render")
        gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "render_gt")
        diffuse_render_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "diffuse")
        diffuse_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "diffuse_gt")
        glossy_render_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "glossy")
        glossy_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "glossy_gt")
        depth_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "depth")
        depth_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "depth_gt")
        normal_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "normal")
        normal_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "normal_gt")
        roughness_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "roughness")
        roughness_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "roughness_gt")
        f0_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "f0")
        f0_gts_path = os.path.join(cli.model_path, split, "ours_{}".format(iteration), "f0_gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(diffuse_render_path, exist_ok=True)
        makedirs(diffuse_gts_path, exist_ok=True)
        makedirs(glossy_render_path, exist_ok=True)
        makedirs(glossy_gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)
        makedirs(depth_gts_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(normal_gts_path, exist_ok=True)
        makedirs(roughness_path, exist_ok=True)
        makedirs(roughness_gts_path, exist_ok=True)
        makedirs(f0_path, exist_ok=True)
        makedirs(f0_gts_path, exist_ok=True)

        all_renders = []
        all_gts = []

        all_diffuse_renders = []
        all_diffuse_gts = []

        all_glossy_renders = []
        all_glossy_gts = []

        all_depth_renders = []
        all_depth_gts = []

        all_normal_renders = []
        all_normal_gts = []

        all_roughness_renders = []
        all_roughness_gts = []

        all_f0_renders = []
        all_f0_gts = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if "env" in mode:
                if idx == 0:
                    view0 = view
                    view0.FoVx = 2.0944 * 2
                    view0.FoVy = -2.0944 * 2
                    continue  # * Skip frame 0, rotation is incorrect
                view = view0

                R_colmap_init = view.R
                _R_blender = -R_colmap_init
                _R_blender[:, 0] = -_R_blender[:, 0]
                R_blender = _R_blender
                T_blender = -R_colmap_init @ view.T

                if "env_rot" in mode:
                    theta = (2 * math.pi * idx) / len(views)
                    rotation = torch.tensor(
                        (
                            (math.cos(theta), -math.sin(theta), 0.0),
                            (math.sin(theta), math.cos(theta), 0.0),
                            (0.0, 0.0, 1.0),
                        )
                    )
                    if idx > 0:
                        R_blender = rotation.to(torch.float64) @ np.array(
                            (
                                (
                                    -0.9882196187973022,
                                    0.10767492651939392,
                                    -0.10875695198774338,
                                ),
                                (
                                    -0.10844696313142776,
                                    0.008747747167944908,
                                    0.9940638542175293,
                                ),
                                (
                                    0.10798710584640503,
                                    0.994147777557373,
                                    0.003032323671504855,
                                ),
                            )
                        )
                elif "env_move" in mode:
                    theta = 0
                    rotation = torch.tensor(
                        (
                            (math.cos(theta), -math.sin(theta), 0.0),
                            (math.sin(theta), math.cos(theta), 0.0),
                            (0.0, 0.0, 1.0),
                        )
                    )
                    R_blender = rotation.to(torch.float64) @ np.array(
                        (
                            (
                                -0.9882196187973022,
                                0.10767492651939392,
                                -0.10875695198774338,
                            ),
                            (
                                -0.10844696313142776,
                                0.008747747167944908,
                                0.9940638542175293,
                            ),
                            (
                                0.10798710584640503,
                                0.994147777557373,
                                0.003032323671504855,
                            ),
                        )
                    )

                if mode == "env_rot_1":
                    T_blender = np.array([0.0, -0.2, 0.2])
                elif mode == "env_rot_2":
                    T_blender = np.array([1.3, -2.0, 0.0])
                elif mode == "env_move_1":
                    t = idx / (len(views) - 1)
                    T_blender = (1.0 - t) * np.array([0.0, -0.2, 0.2]) + t * np.array([1.3, -2.0, 0.0])
                elif mode == "env_move_2":
                    t = idx / (len(views) - 1)
                    T_blender = (1.0 - t) * np.array([0.0, -0.2, 0.2]) + t * np.array([1.3, -0.3, 0.0])

                R_colmap = -R_blender
                R_colmap[:, 0] = -R_colmap[:, 0]
                T_colmap = -R_colmap.T @ T_blender

                view.R = np.array(R_colmap)
                view.T = np.array(T_colmap)

                view.update()

            config = raytracer.cuda_module.get_config()

            if cli.spp > 1:
                config.accumulate_samples.copy_(True)
                raytracer.cuda_module.reset_accumulators()
                for _ in range(cli.spp):
                    package = render(
                        view,
                        raytracer,
                        denoise=False,
                        znear=cli.znear,
                    ) 
                if cli.denoise:
                    raytracer.cuda_module.denoise()
                    package.final = raytracer.cuda_module.get_framebuffer().output_denoised.clone().detach().moveaxis(-1, 1)
            else:
                package = render(
                    view,
                    raytracer,
                    denoise=cli.denoise,
                    znear=cli.znear,
                )

            diffuse_gt_image = tonemap(view.diffuse_image).clamp(0.0, 1.0)
            glossy_gt_image = tonemap(view.glossy_image).clamp(0.0, 1.0)
            gt_image = tonemap(view.original_image).clamp(0.0, 1.0)
            normal_gt_image = view.normal_image
            roughness_gt_image = view.roughness_image
            depth_gt_image = view.depth_image.unsqueeze(0)
            f0_gt_image = view.f0_image

            diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
            glossy_image = tonemap(package.rgb[1:].sum(dim=0)).clamp(0, 1)
            pred_image = tonemap(package.final.squeeze(0)).clamp(0, 1)

            if not cli.skip_save_frames and mode == "regular":
                torchvision.utils.save_image(
                    glossy_image,
                    os.path.join(glossy_render_path, "{0:05d}".format(idx) + "_glossy.png"),
                )
                torchvision.utils.save_image(
                    glossy_gt_image,
                    os.path.join(glossy_gts_path, "{0:05d}".format(idx) + "_glossy.png"),
                )

                torchvision.utils.save_image(
                    diffuse_image,
                    os.path.join(diffuse_render_path, "{0:05d}".format(idx) + "_diffuse.png"),
                )
                torchvision.utils.save_image(
                    diffuse_gt_image,
                    os.path.join(diffuse_gts_path, "{0:05d}".format(idx) + "_diffuse.png"),
                )

                torchvision.utils.save_image(
                    package.depth[0].unsqueeze(0) / package.target_depth.amax(),
                    os.path.join(depth_path, "{0:05d}".format(idx) + "_depth.png"),
                )
                torchvision.utils.save_image(
                    depth_gt_image / package.target_depth.amax(),
                    os.path.join(depth_gts_path, "{0:05d}".format(idx) + "_depth.png"),
                )

                torchvision.utils.save_image(
                    package.normal[0] / 2 + 0.5,
                    os.path.join(normal_path, "{0:05d}".format(idx) + "_normal.png"),
                )
                torchvision.utils.save_image(
                    normal_gt_image / 2 + 0.5,
                    os.path.join(normal_gts_path, "{0:05d}".format(idx) + "_normal.png"),
                )

                torchvision.utils.save_image(
                    package.roughness[0],
                    os.path.join(roughness_path, "{0:05d}".format(idx) + "_roughness.png"),
                )
                torchvision.utils.save_image(
                    roughness_gt_image,
                    os.path.join(roughness_gts_path, "{0:05d}".format(idx) + "_roughness.png"),
                )

                torchvision.utils.save_image(
                    package.f0[0],
                    os.path.join(f0_path, "{0:05d}".format(idx) + "_f0.png"),
                )
                torchvision.utils.save_image(
                    f0_gt_image,
                    os.path.join(f0_gts_path, "{0:05d}".format(idx) + "_f0.png"),
                )

                torchvision.utils.save_image(
                    gt_image,
                    os.path.join(gts_path, "{0:05d}".format(idx) + "_render.png"),
                )
                torchvision.utils.save_image(
                    pred_image,
                    os.path.join(render_path, "{0:05d}".format(idx) + "_render.png"),
                )

            def format_image(image):
                # * Enforce even dimensions for video encoding
                rounded_size = (image.shape[-2] // 2 * 2, image.shape[-1] // 2 * 2) 
                if rounded_size != (image.shape[-2], image.shape[-1]):
                    image = F.interpolate(
                        image[None],
                        (image.shape[-2] // 2 * 2, image.shape[-1] // 2 * 2),
                        mode="bilinear",
                    )[0]
                return (image.clamp(0, 1) * 255).to(torch.uint8).moveaxis(0, -1).cpu()

            all_renders.append(format_image(pred_image))
            all_gts.append(format_image(tonemap(package.target)))

            all_diffuse_renders.append(format_image(diffuse_image))
            all_diffuse_gts.append(format_image(tonemap(package.target_diffuse)))

            all_glossy_renders.append(format_image(glossy_image))
            all_glossy_gts.append(format_image(tonemap(package.target_glossy)))

            max_depth = package.target_depth.amax()
            all_depth_renders.append(format_image(package.depth[0] / max_depth).repeat(1, 1, 3))
            all_depth_gts.append(format_image(package.target_depth / max_depth).repeat(1, 1, 3))

            all_normal_renders.append(format_image(package.normal[0] / 2 + 0.5))
            all_normal_gts.append(format_image(package.target_normal / 2 + 0.5))

            all_roughness_renders.append(format_image(package.roughness[0].repeat(3, 1, 1)))
            all_roughness_gts.append(format_image(package.target_roughness.repeat(3, 1, 1)))

            all_f0_renders.append(format_image(package.f0[0]))
            all_f0_gts.append(format_image(package.target_f0))

        video_dir = os.path.join("videos", mode)
        os.makedirs(os.path.join(cli.model_path, video_dir), exist_ok=True)

        if not cli.skip_video:
            print("Writing videos...")
            path = os.path.join(cli.model_path, f"{{dir}}", f"{split}_{{name}}.mp4")

            kwargs = {"fps": 30, "options":{"crf": "30"}}

            torchvision.io.write_video(
                path.format(name=f"final", dir=video_dir),
                torch.cat([torch.stack(all_renders), torch.stack(all_gts)], dim=2),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"diffuse", dir=video_dir),
                torch.cat(
                    [
                        torch.stack(all_diffuse_renders),
                        torch.stack(all_diffuse_gts),
                    ],
                    dim=2,
                ),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"glossy", dir=video_dir),
                torch.cat(
                    [
                        torch.stack(all_glossy_renders),
                        torch.stack(all_glossy_gts),
                    ],
                    dim=2,
                ),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"depth", dir=video_dir),
                torch.cat(
                    [
                        torch.stack(all_depth_renders),
                        torch.stack(all_depth_gts),
                    ],
                    dim=2,
                ),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"normal", dir=video_dir),
                torch.cat(
                    [
                        torch.stack(all_normal_renders),
                        torch.stack(all_normal_gts),
                    ],
                    dim=2,
                ),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"roughness", dir=video_dir),
                torch.cat(
                    [
                        torch.stack(all_roughness_renders),
                        torch.stack(all_roughness_gts),
                    ],
                    dim=2,
                ),
                **kwargs,
            )
            torchvision.io.write_video(
                path.format(name=f"f0", dir=video_dir),
                torch.cat(
                    [torch.stack(all_f0_renders), torch.stack(all_f0_gts)],
                    dim=2,
                ),
                **kwargs,
            )


if __name__ == "__main__":
    cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)
    saved_cli_path = os.path.join(cli.model_path, "cfg.json")
    cfg = tyro.cli(TyroConfig, args=unknown_args, default=TyroConfig(**json.load(open(saved_cli_path, "r"))))
    
    set_seeds()

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=cli.iteration, shuffle=False, model_path=cli.model_path)

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height)

    if cli.split == "train":
        render_set(
            cli,
            cfg,
            scene,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            raytracer,
        )
    else:
        render_set(
            cli,
            cfg,
            scene,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            raytracer,
        )
