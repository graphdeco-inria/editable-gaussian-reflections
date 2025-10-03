# ruff: noqa
import random
import sys

import torch
from torchvision.io import write_video
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append("splatting_code")
# from utils.loss_utils import l1_loss as l1_loss_gs, ssim as ssim_gs

build_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
print("Build dir:", build_dir)
torch.classes.load_library(f"{build_dir}/libgausstracer.so")

import sys

sys.path.append(build_dir)
import os

import numpy as np
import torch.nn as nn
from load_camera_poses import *
from plyfile import PlyData

if LOAD_FROM_PLY := False:
    plydata = PlyData.read("data/nocluster.ply")

    xyz = torch.from_numpy(
        np.vstack(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]
        ).T
    )
    opacity = torch.from_numpy(plydata["vertex"]["opacity"])
    scale = torch.from_numpy(
        np.vstack(
            [
                plydata["vertex"]["scale_0"],
                plydata["vertex"]["scale_1"],
                plydata["vertex"]["scale_2"],
            ]
        ).T
    )
    rotation = torch.from_numpy(
        np.vstack(
            [
                plydata["vertex"]["rot_0"],
                plydata["vertex"]["rot_1"],
                plydata["vertex"]["rot_2"],
                plydata["vertex"]["rot_3"],
            ]
        ).T
    )
    features = torch.from_numpy(
        np.vstack([plydata["vertex"][f"f_dc_{i}"] for i in range(3)])
    ).T

    position = torch.from_numpy(
        np.vstack(
            [
                plydata["vertex"]["pos_0"],
                plydata["vertex"]["pos_1"],
                plydata["vertex"]["pos_2"],
            ]
        ).T
    )
    normal = torch.from_numpy(
        np.vstack(
            [
                plydata["vertex"]["normal_0"],
                plydata["vertex"]["normal_1"],
                plydata["vertex"]["normal_2"],
            ]
        ).T
    )
    f0 = torch.from_numpy(
        np.vstack(
            [
                plydata["vertex"]["f0_0"],
                plydata["vertex"]["f0_1"],
                plydata["vertex"]["f0_2"],
            ]
        ).T
    )
    roughness = torch.from_numpy(plydata["vertex"]["roughness"])

    gaussian_means = xyz.contiguous().float().cuda()
    gaussian_opacity = opacity.contiguous().float().cuda().unsqueeze(1)
    gaussian_scales = scale.contiguous().float().cuda()
    gaussian_rotations = rotation.contiguous().float().cuda()
    gaussian_features = features.contiguous().float().cuda()
    gaussian_rgb = (
        (features[:, :3]).contiguous().float().cuda()
    )  # * (C0 := 0.28209479177387814) + 0.5, 0.0).contiguous().float().cuda()
    gaussian_positions = position.contiguous().float().cuda()
    gaussian_normals = normal.contiguous().float().cuda()
else:
    max = None
    data = {**torch.load("data/raytrace_data.pt"), **torch.load("data/extras.pt")}
    gaussian_means = data["xyz"].contiguous()
    gaussian_means.grad = torch.zeros_like(gaussian_means)
    gaussian_opacity = data["opacity"].contiguous()
    gaussian_opacity.grad = torch.zeros_like(gaussian_opacity)
    gaussian_scales = data["scale"].contiguous()
    gaussian_scales.grad = torch.zeros_like(gaussian_scales)
    gaussian_rotations = data["rotation"].contiguous()
    gaussian_rotations.grad = torch.zeros_like(gaussian_rotations)
    gaussian_features = data["features"].contiguous()
    gaussian_features.grad = torch.zeros_like(gaussian_features)
    gaussian_rgb = torch.clamp_min(
        data["features"][:, 0] * (C0 := 0.28209479177387814) + 0.5, 0.0
    )
    gaussian_rgb.grad = torch.zeros_like(gaussian_rgb)
    gaussian_positions = torch.zeros_like(gaussian_means)
    gaussian_positions.grad = torch.zeros_like(gaussian_means)
    gaussian_normals = torch.zeros_like(gaussian_means)
    gaussian_normals.grad = torch.zeros_like(gaussian_means)

# -----------------------------------------------------------------------------------------------------------------

VIDEO = True

# example_target_rgb = TF.to_tensor(Image.open("data/render_splat.png").convert("RGB")).moveaxis(0, -1).to("cuda").contiguous()
num_steps = int(os.getenv("ITERATIONS", 4000))

if not LOAD_FROM_PLY:
    # inverse exp the scales
    gaussian_scales.data.copy_(torch.log(gaussian_scales.data))

if "SPLAT" in os.environ:
    # inverse sigmoid the rgb
    gaussian_rgb.data.copy_(torch.log(gaussian_rgb.data / (1.0 - gaussian_rgb.data)))
else:
    # inverse softplus the rgb
    gaussian_rgb.data.copy_(torch.log(torch.exp(gaussian_rgb.data) - 1.0))

if not LOAD_FROM_PLY:
    # inverse sigmoid the opacity
    gaussian_opacity.data.copy_(
        torch.log(gaussian_opacity.data / (1.0 - gaussian_opacity.data))
    )

if "SCENE" in os.environ:
    scene = torch.load(os.environ["SCENE"])
    with torch.no_grad():
        gaussian_opacity.copy_(scene["opacity"])
        gaussian_rgb.copy_(scene["features"].unsqueeze(1))
        gaussian_means.copy_(scene["xyz"])
        gaussian_scales.copy_(scene["scale"])
        gaussian_rotations.copy_(scene["rotation"])

if False:
    if "SPLAT" not in os.environ:
        print(
            "Percent of gaussians with opacity values under 0.01:",
            (torch.sigmoid(gaussian_opacity) < raytracer_config.ALPHA_THRESHOLD)
            .float()
            .mean(),
        )
        with torch.no_grad():
            mask = (
                torch.sigmoid(gaussian_opacity) >= raytracer_config.ALPHA_THRESHOLD
            ).flatten()
            gaussian_opacity.data = gaussian_opacity[mask].requires_grad_()
            gaussian_rgb.data = gaussian_rgb[mask].requires_grad_()
            gaussian_means.data = gaussian_means[mask].requires_grad_()
            gaussian_rotations.data = gaussian_rotations[mask].requires_grad_()
            gaussian_scales.data = gaussian_scales[mask].requires_grad_()

if "SPLAT" in os.environ:
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    pc = GaussianModel(3 if "SH" in os.environ else 0)
    pc._xyz = nn.Parameter(gaussian_means)
    pc._features_dc = nn.Parameter(gaussian_rgb.unsqueeze(1))
    pc._features_rest = nn.Parameter(gaussian_rgb.unsqueeze(1)[:, 0:0])
    if "SH" in os.environ:
        pc.active_sh_degree = 3
        features = torch.zeros((gaussian_rgb.shape[0], 3, (3 + 1) ** 2)).float().cuda()
        pc._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous()
        )
    pc._scaling = nn.Parameter(gaussian_scales)
    pc._rotation = nn.Parameter(gaussian_rotations)
    pc._opacity = nn.Parameter(gaussian_opacity)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("build_name", type=str, default="build")
parser.add_argument("output_name", type=str, default="output/default")
parser.add_argument("-t", type=float, default=0.001)
parser.add_argument("-a", type=float, default=0.01)
parser.add_argument("-e", type=float, default=2)
parser.add_argument("-d", type=float, default=0.0)
parser.add_argument("-y", type=float, default=0.333)
parser.add_argument("-p", type=float, default=None)
conf = parser.parse_args()

os.makedirs(conf.output_name, exist_ok=True)
os.makedirs(conf.output_name.replace("output/", "checkpoints/"), exist_ok=True)
os.chdir(conf.output_name)

# -----------------------------------------------------------------------------------------------------------------

raytracer = torch.classes.gausstracer.Raytracer(
    IMAGE_WIDTH, IMAGE_HEIGHT, gaussian_rgb.shape[0], 300_000_000, 200_000_000
)

gaussians = raytracer.get_gaussians()
with torch.no_grad():
    gaussians.rgb.copy_(gaussian_rgb)
    gaussians.opacity.copy_(gaussian_opacity)
    gaussians.scale.copy_(gaussian_scales)
    gaussians.rotation.copy_(gaussian_rotations)
    gaussians.mean.copy_(gaussian_means)

raytracer.rebuild_bvh()
raytracer.raytrace()

#!!
# raytracer.configure(conf.t, conf.a, conf.e)  # -e 12 drops it down to 15ms after 200 iters

# -----------------------------------------------------------------------------------------------------------------


if "SPLAT" in os.environ:
    adam = torch.optim.Adam(
        [
            dict(params=[pc._features_dc], lr=0.025),
            dict(params=[pc._opacity], lr=0.025),
            dict(
                params=[pc._xyz], lr=0.000016
            ),  # was off due to a typo (0.0001/6), vs 0.00016 and psnr was better, now i'm at 0.000016 (one extra zero)
            dict(params=[pc._rotation], lr=0.001),
            dict(params=[pc._scaling], lr=0.005),  # was too high at 0.05
        ]
        + (
            [dict(params=[pc._features_rest], lr=0.025 / 20)]
            if "SH" in os.environ
            else []
        ),
        betas=[0.9, 0.999],
        eps=1e-15,
    )
else:
    adam = torch.optim.Adam(
        [
            dict(params=[gaussians.rgb], lr=0.025),
            dict(params=[gaussians.opacity], lr=0.025),
            dict(params=[gaussians.mean], lr=0.000016),
            dict(params=[gaussians.rotation], lr=0.001),
            dict(
                params=[gaussians.scale], lr=0.005
            ),  # was too high at 0.05 while gs proposed lr is 0.005
        ],
        betas=[0.9, 0.999],
        eps=1e-15,
    )


checkpoint_iters = [4000, 8000, 16000, 32000]
if num_steps not in checkpoint_iters:
    checkpoint_iters.append(num_steps)
checkpoints = {}

gaussian_rgb_ema = gaussian_rgb.clone().detach()
gaussian_opacity_ema = gaussian_opacity.clone().detach()
gaussian_means_ema = gaussian_means.clone().detach()
gaussian_rotations_ema = gaussian_rotations.clone().detach()
gaussian_scales_ema = gaussian_scales.clone().detach()

init_lrs = {}


def splat_render(cam_blender, target=None, start_event=None, end_event=None):
    with torch.no_grad():
        R_colmap = -cam_blender[:3, :3].clone()
        R_colmap[:, 0] = -R_colmap[:, 0]
        R_colmap = np.array(R_colmap)
        T_colmap = np.array(cam_blender[:3, 3].clone()) @ -R_colmap

        viewpoint_cam = Camera(
            0,
            R_colmap,
            T_colmap,
            0.6911121570925639,
            0.4710906705955732,
            torch.zeros(3, 1024, 1536),
            None,
            None,
            None,
        )
    torch.cuda.synchronize()
    if start_event is not None:
        start_event.record()
    package = render(viewpoint_cam, pc, torch.tensor([0.0, 0.0, 0.0], device="cuda"))
    if target is not None:
        image = package["render"].moveaxis(0, -1)
        loss = torch.nn.functional.mse_loss(image, target)
        loss.backward()
    else:
        loss = None
    if end_event is not None:
        end_event.record()
    torch.cuda.synchronize()

    with torch.no_grad():
        output_rgb.copy_(package["render"].moveaxis(0, -1))

    return loss


frames = []
frames_t = []
losses = []
avg_scales = []
avg_volume = []
avg_opacity = []
frame_times = []
iter_times = []
pbar = tqdm(range(num_steps), desc="Optimizing")


def prepare_checkpoint():
    return {
        "rgb": gaussian_rgb_ema.detach().cpu().clone(),
        "opacity": gaussian_opacity_ema.detach().cpu().clone(),
        "means": gaussian_means_ema.detach().cpu().clone(),
        "rotations": gaussian_rotations_ema.detach().cpu().clone(),
        "scales": gaussian_scales_ema.detach().cpu().clone(),
    }


@torch.no_grad()
def load_checkpoint(checkpoint):
    if "SPLAT" in os.environ:
        pc._features_dc.data.copy_(checkpoint["rgb"].unsqueeze(1))
        pc._opacity.data.copy_(checkpoint["opacity"])
        pc._xyz.data.copy_(checkpoint["means"])
        pc._rotation.data.copy_(checkpoint["rotations"])
        pc._scaling.data.copy_(checkpoint["scales"])
    else:
        gaussians.rgb.copy_(checkpoint["rgb"])
        gaussians.opacity.copy_(checkpoint["opacity"])
        gaussians.mean.copy_(checkpoint["means"])
        gaussians.rotation.copy_(checkpoint["rotations"])
        gaussians.scale.copy_(checkpoint["scales"])


# -----------------------------------------------------------------------------------------------------------------

framebuffer = raytracer.get_framebuffer()

raytracer.get_config().num_bounces.fill_(0)

for i in pbar:
    adam.zero_grad(set_to_none=False)

    if (
        i == int(1 * num_steps / 4)
        or i == int(2 * num_steps / 4)
        or i == int(3 * num_steps / 4)
    ):
        for group in adam.param_groups:
            for p in group["params"]:
                group["lr"] *= conf.y
                init_lrs[p] = group["lr"]

    j = random.choice(range(NUM_TRAIN_IMAGES))
    cam = train_cam_poses[j]

    framebuffer.target_rgb.copy_(train_images[j])
    framebuffer.target_diffuse.copy_(train_images[j])  #!!!!!!

    camera = raytracer.get_camera()
    camera.znear.copy_(torch.tensor(0.0001, device="cuda"))
    camera.zfar.copy_(torch.tensor(100.0, device="cuda"))
    camera.vertical_fov_radians.copy_(torch.tensor(0.4710899591445923, device="cuda"))
    camera_c2w_rot_blender = cam[:3, :3].contiguous()
    camera_position = cam[:3, 3].contiguous()
    camera.set_pose(camera_position, camera_c2w_rot_blender)
    print(camera_position)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        raytracer.raytrace()
    end_event.record()
    torch.cuda.synchronize()

    if "SPLAT" in os.environ:
        loss = splat_render(cam, target=framebuffer.target_rgb)
    else:
        raytracer.update_bvh()
        raytracer.raytrace()

        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(
                framebuffer.output_rgb[0, ..., :3], framebuffer.target_rgb
            )

    end_event.record()
    torch.cuda.synchronize()
    time_for_iteration = start_event.elapsed_time(end_event)

    with torch.no_grad():
        if i % 10 == 0:
            if "SPLAT" in os.environ:
                avg_scales.append(
                    pc.get_scaling.data.detach().sort(dim=1).values.mean(dim=0).cpu()
                )
                avg_volume.append(pc.get_scaling.data.detach().prod(dim=1).mean().cpu())
                avg_opacity.append(pc.get_opacity.data.detach().mean().cpu())
            else:
                avg_scales.append(
                    torch.exp(gaussian_scales.cpu())
                    .detach()
                    .sort(dim=1)
                    .values.mean(dim=0)
                )
                avg_volume.append(
                    torch.exp(gaussian_scales.cpu()).detach().prod(dim=1).mean()
                )
                avg_opacity.append(
                    torch.sigmoid(gaussian_opacity.cpu()).detach().mean()
                )

            losses.append(loss.item())

            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                if "SPLAT" in os.environ:
                    splat_render(cam, start_event=start_event, end_event=end_event)
                else:
                    start_event.record()
                    raytracer.raytrace()
                    end_event.record()
                    torch.cuda.synchronize()
                time_for_frame = start_event.elapsed_time(end_event)
                frame_times.append(time_for_frame)
                iter_times.append(time_for_iteration)
                ips = 1000 / time_for_iteration
                fps = 1000 / time_for_frame
                pbar.set_postfix(
                    ips=round(ips),
                    fps=round(fps),
                    ms_iter=f"{time_for_iteration:.2f}",
                    ms_frame=f"{time_for_frame:.2f}",
                )

    gaussians.rgb.grad = gaussians.dL_drgb
    gaussians.opacity.grad = gaussians.dL_dopacity
    gaussians.mean.grad = gaussians.dL_dmean
    gaussians.rotation.grad = gaussians.dL_drotation
    gaussians.scale.grad = gaussians.dL_dscale

    adam.step()

    gaussians.dL_drgb.zero_()
    gaussians.dL_dopacity.zero_()
    gaussians.dL_dmean.zero_()
    gaussians.dL_drotation.zero_()
    gaussians.dL_dscale.zero_()

    decay = 0.95
    if "SPLAT" in os.environ:
        gaussian_rgb_ema.mul_(decay).add_(pc._features_dc.data[:, 0], alpha=1 - decay)
        gaussian_opacity_ema.mul_(decay).add_(pc._opacity.data, alpha=1 - decay)
        gaussian_means_ema.mul_(decay).add_(pc._xyz.data, alpha=1 - decay)
        gaussian_rotations_ema.mul_(decay).add_(pc._rotation.data, alpha=1 - decay)
        gaussian_scales_ema.mul_(decay).add_(pc._scaling.data, alpha=1 - decay)
    else:
        gaussian_rgb_ema.data.mul_(decay).add_(gaussians.rgb, alpha=1 - decay)
        gaussian_opacity_ema.data.mul_(decay).add_(gaussians.opacity, alpha=1 - decay)
        gaussian_means_ema.data.mul_(decay).add_(gaussians.mean, alpha=1 - decay)
        gaussian_rotations_ema.data.mul_(decay).add_(
            gaussians.rotation, alpha=1 - decay
        )
        gaussian_scales_ema.data.mul_(decay).add_(gaussians.scale, alpha=1 - decay)

    if i % 3 == 0:
        # cam = test_cam_poses[0]
        # camera_c2w_rot_blender = cam[:3, :3].contiguous()
        # camera_position = cam[:3, 3].contiguous()
        # camera.set_pose(camera_position, camera_c2w_rot_blender)
        # raytracer.update_bvh()

        with torch.no_grad():
            if "SPLAT" in os.environ:
                splat_render(cam)
            else:
                raytracer.raytrace()
            frame = (
                torch.cat(
                    [framebuffer.output_rgb[0, ..., :3], framebuffer.target_rgb], dim=1
                )
                .detach()
                .cpu()
            )
            frame_t = framebuffer.output_transmittance.detach().cpu()
        if VIDEO:
            frames.append(frame)
            frames_t.append(frame_t)

    if i == 0:
        save_image(frame.moveaxis(-1, 0), "optim_start.png")

    if i + 1 in checkpoint_iters or i == num_steps - 1:
        latest_checkpoint = checkpoints[i + 1] = prepare_checkpoint()
        cwd = os.getcwd()
        newcwd = cwd.replace("/output/", "/checkpoints/")
        os.chdir(newcwd)
        torch.save(latest_checkpoint, f"checkpoint_{i + 1}.pt")
        os.chdir(cwd)

    if i % 1000 == 0:
        raytracer.rebuild_bvh()

if num_steps == 0:
    checkpoints[0] = prepare_checkpoint()
else:
    save_image(frame.moveaxis(-1, 0), "optim_end.png")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def create_plot(label, df):
        # Define a smoothing function
        def smooth_curve(series, smoothing_factor):
            if smoothing_factor == 0:
                return series  # No smoothing for 0
            return series.rolling(
                window=int(smoothing_factor), center=True, min_periods=1
            ).mean()

        # Default smoothing factor
        default_smoothing = 20

        figs = []

        for col_name in df.columns:
            smoothed_data = smooth_curve(df[col_name], default_smoothing)

            # Calculate the fixed y-axis range based on the original data
            y_min, y_max = df[col_name].min(), df[col_name].max()

            # Create the figure
            fig = go.Figure()

            # Add only the smoothed col_name data
            fig.add_trace(
                go.Scatter(
                    y=smoothed_data, mode="lines", name=col_name, showlegend=False
                )
            )

            figs.append(fig)

        # Create a subplot grid
        if len(df.columns) == 1:
            fig = make_subplots(rows=1, cols=1, subplot_titles=df.columns)
            for trace in figs[0]["data"]:
                fig.add_trace(trace, row=1, col=1)
        elif len(df.columns) == 4:
            fig = make_subplots(rows=2, cols=2, subplot_titles=df.columns)
            for i, subfig in enumerate(figs):
                for trace in subfig["data"]:
                    fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)
        else:
            fig = make_subplots(rows=2, cols=3, subplot_titles=df.columns)
            for i, subfig in enumerate(figs):
                for trace in subfig["data"]:
                    fig.add_trace(trace, row=(i // 3) + 1, col=(i % 3) + 1)

        # Add a slider to adjust the smoothing factor
        steps = []
        for smoothing_factor in range(
            0, 101, 1
        ):  # Slider values from 0 to 100 in increments of 1
            step = dict(
                method="update",
                args=[
                    {
                        "y": [
                            smooth_curve(df[col_name], smoothing_factor)
                            for col_name in df.columns
                        ]
                    },  # Update all traces
                ],
                label=str(smoothing_factor),
            )
            steps.append(step)

        sliders = [
            dict(
                active=default_smoothing,
                currentvalue={"prefix": "Smoothing Factor: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        # Update layout
        fig.update_layout(
            sliders=sliders,
            template="plotly_dark",
            title="Training Stats",
            xaxis_title="Index",
            yaxis_title="Value",
            height=800,
            width=1200,
            paper_bgcolor="rgb(17, 17, 17)",
        )

        # Save the combined figure to an HTML file with custom CSS
        fig.write_html(f"plot_{label}.html", include_plotlyjs="cdn", full_html=False)
        with open(f"plot_{label}.html", "a") as f:
            f.write("""
<style>
    body {
        margin: 0;
        background: rgb(17, 17, 17);
    }
</style>
""")

    create_plot(
        "stats",
        pd.DataFrame(
            {
                "loss": losses,
                "volume": avg_volume,
                "opacity": avg_opacity,
                "min_scales": torch.stack(avg_scales)[:, 0].cpu().numpy(),
                "median_scales": torch.stack(avg_scales)[:, 1].cpu().numpy(),
                "max_scales": torch.stack(avg_scales)[:, 2].cpu().numpy(),
            }
        ),
    )
    create_plot(
        "losses",
        pd.DataFrame(
            {
                "loss": losses,
            }
        ),
    )
    create_plot(
        "times",
        pd.DataFrame(
            {
                "ms/frame": frame_times,
                "ms/iter": iter_times,
                "FPS": [1000 / x for x in frame_times],
                "ITS": [1000 / x for x in iter_times],
            }
        ),
    )

if VIDEO:
    write_video(
        "train.mp4", torch.stack(frames).clamp(0, 1) * 255, 30, options={"crf": "28"}
    )  # 15 for good quality
    # write_video(f"train_t.mp4", torch.cat([torch.stack(frames_t)[..., 0].unsqueeze(-1).repeat(1, 1, 1, 3), torch.stack(frames_t)[..., 1].unsqueeze(-1).repeat(1, 1, 1, 3)], dim=2).clamp(0, 1) * 255, 30, options={ "crf": "28" }) #15 for good quality

# ---------------------------------------------------------------------

import sys

from plotly.subplots import make_subplots
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class DualOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


psnr = PeakSignalNoiseRatio(data_range=1.0).cpu()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cpu()
lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True)


def run_eval(split_name, iter_count, dataset_cam_poses, dataset_images):
    raytracer.rebuild_bvh()

    our_frames = []
    gt_frames = []
    frames = []

    avg_time = 0.0
    avg_iteration_time = 0.0

    gaussian_rgb_avg_grad_norm = torch.zeros_like(gaussian_rgb)
    gaussian_opacity_avg_grad_norm = torch.zeros_like(gaussian_opacity)
    gaussian_means_avg_grad_norm = torch.zeros_like(gaussian_means)
    gaussian_rotations_avg_grad_norm = torch.zeros_like(gaussian_rotations)
    gaussian_scales_avg_grad_norm = torch.zeros_like(gaussian_scales)

    for i, cam in enumerate(tqdm(dataset_cam_poses, desc="Evaluating " + split_name)):
        camera_c2w_rot_blender = cam[:3, :3].contiguous()
        camera_position = cam[:3, 3].contiguous()
        camera.set_pose(camera_position, camera_c2w_rot_blender)
        #!!!
        framebuffer.target_rgb.copy_(dataset_images[i])
        framebuffer.target_diffuse.copy_(dataset_images[i])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            if "SPLAT" in os.environ:
                splat_render(cam, start_event=start_event, end_event=end_event)
            else:
                start_event.record()
                raytracer.update_bvh()
                raytracer.raytrace()
                end_event.record()
        torch.cuda.synchronize()

        our_frames.append(framebuffer.output_rgb[0, ..., :3].clone().cpu())
        gt_frames.append(framebuffer.target_rgb.cpu())

        frames.append(
            torch.cat(
                [framebuffer.output_rgb[0, ..., :3], framebuffer.target_rgb], dim=0
            )
            .detach()
            .cpu()
        )
        frame_time = start_event.elapsed_time(end_event)
        avg_time += frame_time

        adam.zero_grad(set_to_none=False)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        if "SPLAT" in os.environ:
            splat_render(
                cam, target=target_rgb, start_event=start_event, end_event=end_event
            )
        else:
            start_event.record()
            raytracer.update_bvh()
            raytracer.raytrace()
            end_event.record()

        if "SPLAT" in os.environ:
            gaussian_rgb_avg_grad_norm += pc._features_dc.grad.squeeze(1).abs()
            gaussian_opacity_avg_grad_norm += pc._opacity.grad.abs()
            gaussian_means_avg_grad_norm += pc._xyz.grad.abs()
            gaussian_rotations_avg_grad_norm += pc._rotation.grad.abs()
            gaussian_scales_avg_grad_norm += pc._scaling.grad.abs()
        else:
            gaussian_rgb_avg_grad_norm += gaussians.rgb.grad.abs()
            gaussian_opacity_avg_grad_norm += gaussians.opacity.grad.abs()
            gaussian_means_avg_grad_norm += gaussians.mean.grad.abs()
            gaussian_rotations_avg_grad_norm += gaussians.rotation.grad.abs()
            gaussian_scales_avg_grad_norm += gaussians.scale.grad.abs()

        torch.cuda.synchronize()

        iter_time = start_event.elapsed_time(end_event)
        avg_iteration_time += iter_time

    avg_time /= len(dataset_cam_poses)
    avg_iteration_time /= len(dataset_cam_poses)

    gaussian_rgb_avg_grad_norm /= len(dataset_cam_poses)
    gaussian_opacity_avg_grad_norm /= len(dataset_cam_poses)
    gaussian_means_avg_grad_norm /= len(dataset_cam_poses)
    gaussian_rotations_avg_grad_norm /= len(dataset_cam_poses)
    gaussian_scales_avg_grad_norm /= len(dataset_cam_poses)

    with open(f"{split_name}_results.txt", "a") as log_file:
        print(
            "==========================================================================="
        )

        dual_output = DualOutput(log_file)
        sys.stdout = dual_output

        print(f"{iter_count} ITERS {split_name.upper()} EVALUATION RESULTS")

        if False:
            print(
                "---------------------------------------------------------------------------"
            )
            # Print grad stats, mean, std in scientific notation with three decimal places

            def round_scientific(value, digits):
                return f"{value:.{digits}e}"

            print(
                "      RGB grad norm mean:",
                round_scientific(gaussian_rgb_avg_grad_norm.mean(), 3),
                "| std:",
                round_scientific(gaussian_rgb_avg_grad_norm.std(), 3),
            )
            print(
                "  Opacity grad norm mean:",
                round_scientific(gaussian_opacity_avg_grad_norm.mean(), 3),
                "| std:",
                round_scientific(gaussian_opacity_avg_grad_norm.std(), 3),
            )
            print(
                "    Means grad norm mean:",
                round_scientific(gaussian_means_avg_grad_norm.mean(), 3),
                "| std:",
                round_scientific(gaussian_means_avg_grad_norm.std(), 3),
            )
            print(
                "Rotations grad norm mean:",
                round_scientific(gaussian_rotations_avg_grad_norm.mean(), 3),
                "| std:",
                round_scientific(gaussian_rotations_avg_grad_norm.std(), 3),
            )
            print(
                "   Scales grad norm mean:",
                round_scientific(gaussian_scales_avg_grad_norm.mean(), 3),
                "| std:",
                round_scientific(gaussian_scales_avg_grad_norm.std(), 3),
            )

        print(
            "---------------------------------------------------------------------------"
        )

        # Calculate PSNR, SSIM, and LPIPS
        psnr_score = psnr(
            torch.stack(our_frames).moveaxis(-1, 1),
            torch.stack(gt_frames).moveaxis(-1, 1),
        ).item()

        psnr_score_clamped = psnr(
            torch.stack(our_frames).moveaxis(-1, 1).clamp(0, 1),
            torch.stack(gt_frames).moveaxis(-1, 1).clamp(0, 1),
        ).item()
        l1_score = torch.nn.functional.l1_loss(
            torch.stack(our_frames).moveaxis(-1, 1).clamp(0, 1),
            torch.stack(gt_frames).moveaxis(-1, 1).clamp(0, 1),
        ).item()
        ssim_score = ssim(
            torch.stack(our_frames).moveaxis(-1, 1).clamp(0, 1),
            torch.stack(gt_frames).moveaxis(-1, 1),
        ).item()
        lpips_score = (
            lpips_fn(
                torch.stack(our_frames).moveaxis(-1, 1).clamp(0, 1),
                torch.stack(gt_frames).moveaxis(-1, 1),
            )
            .mean()
            .item()
        )
        print(
            f"PSNR: {round(psnr_score, 2)}, L1: {round(l1_score, 4)}, SSIM: {round(ssim_score, 4):}, LPIPS: {round(lpips_score, 4)}"
        )
        print(
            f"FPS: {round(1000.0 / avg_time)} ({round(avg_time, 2)}ms/frame), IPS: {round(1000.0 / avg_iteration_time)} ({round(avg_iteration_time, 2)}ms/iter)"
        )

        diff_first_frame = torch.abs(our_frames[0] - gt_frames[0])
        save_image(
            torch.stack(
                [
                    our_frames[0],
                    gt_frames[0],
                    diff_first_frame,
                    diff_first_frame / diff_first_frame.std() / 6,
                ],
                dim=0,
            ).moveaxis(-1, 1),
            "test_first_frame.png",
            nrow=2,
        )
        diff_last_frame = torch.abs(our_frames[-1] - gt_frames[-1])
        save_image(
            torch.stack(
                [
                    our_frames[-1],
                    gt_frames[-1],
                    diff_last_frame,
                    diff_last_frame / diff_last_frame.std() / 6,
                ],
                dim=0,
            ).moveaxis(-1, 1),
            "test_last_frame.png",
            nrow=2,
        )

        write_video(
            f"{split_name}_eval.mp4",
            torch.stack(frames).clamp(0, 1) * 255,
            30,
            options={"crf": "28"},
        )

        print(
            "==========================================================================="
        )
        sys.stdout = dual_output.terminal


if False:
    print("===========================================================================")
    for iter_count, checkpoint in sorted(checkpoints.items(), key=lambda x: x[0]):
        print(f"Loading checkpoint at {iter_count} iterations")
        load_checkpoint(checkpoint)
        run_eval("train", iter_count, train_cam_poses, train_images)
    print("===========================================================================")
print("===========================================================================")
for iter_count, checkpoint in sorted(checkpoints.items(), key=lambda x: x[0]):
    print(f"Loading checkpoint at {iter_count} iterations")
    load_checkpoint(checkpoint)
    run_eval("test", iter_count, test_cam_poses, test_images)

# ---------------------------------------------------------------------------------

# err = False
# if raytracer.gaussian_opacity.isnan().any():
#     print("NAN IN OPACITY!")
#     print("Num nan in opacity:", torch.isnan(raytracer.gaussian_opacity).float().sum())
#     err = True
# if raytracer.gaussian_rgb.isnan().any():
#     print("NAN IN RGB!")
#     print("Num nan in rgb:", torch.isnan(raytracer.gaussian_rgb).float().sum())
#     err = True
# if raytracer.gaussian_scales.isnan().any():
#     print("NAN IN SCALES!")
#     print("Num nan in scales:", torch.isnan(raytracer.gaussian_scales).float().sum())
#     err = True
# if raytracer.gaussian_rotations.isnan().any():
#     print("NAN IN ROTATIONS!")
#     print("Num nan in rotations:", torch.isnan(raytracer.gaussian_rotations).float().sum())
#     err = True
# if raytracer.gaussian_means.isnan().any():
#     print("NAN IN MEANS!")
#     print("Num nan in means:", torch.isnan(raytracer.gaussian_means).float().sum())
#     err = True
# if raytracer.gaussian_opacity.grad.isnan().any():
#     print("NAN IN OPACITY GRAD!")
#     err = True
# if raytracer.gaussian_rgb.grad.isnan().any():
#     print("NAN IN RGB GRAD!")
#     err = True
# if raytracer.gaussian_scales.grad.isnan().any():
#     print("NAN IN SCALES GRAD!")
#     err = True
# if raytracer.gaussian_rotations.grad.isnan().any():
#     print("NAN IN ROTATIONS GRAD!")
#     err = True
# if raytracer.gaussian_means.grad.isnan().any():
#     print("NAN IN MEANS GRAD!")
#     err = True
# # if raytracer.gaussian_exp_power.grad.isnan().any():
# #     print("NAN IN EXP POWER GRAD!")
# #     err = True
# if not err:
#     print("No nans found")

# result = {
#     "opacity": gaussian_opacity,
#     "features": gaussian_rgb,
#     "xyz": gaussian_means,
#     "scale": gaussian_scales,
#     "rotation": gaussian_rotations,
# }
