import os

import torch
import tyro
from tqdm import tqdm
import json

from editable_gauss_refl.config import Config
from editable_gauss_refl.renderer import GaussianRaytracer, render
from editable_gauss_refl.scene import GaussianModel, Scene
from editable_gauss_refl.utils.general_utils import set_seeds
from editable_gauss_refl.utils.system_utils import searchForMaxIteration

from dataclasses import dataclass, field
from typing import Literal, Optional
from typing_extensions import Annotated
from tyro.conf import arg


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Optional[int] = None
    split: Literal["train", "test"] = "test"


@torch.no_grad()
def measure_fps(
    cfg,
    views,
    raytracer,
):
    # * Warmup caches
    for view in tqdm(views, desc="Warmup progress"):
        render(view, raytracer, force_update_bvh=False, targets_available=False, denoise=False)
    
    # * Run measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for view in tqdm(views, desc="Rendering progress"):
        render(view, raytracer, force_update_bvh=False, targets_available=False, denoise=False)
    end_event.record()
    torch.cuda.synchronize()

    # * Output results
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_sec = elapsed_ms / 1000.0
    fps = len(views) / elapsed_sec
    print(f"{fps:.2f} FPS")
    with open(os.path.join(cfg.model_path, "fps.json"), "w") as f:
        f.write(f"{fps:.2f}\n")


if __name__ == "__main__":
    cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)
    saved_cli_path = os.path.join(cli.model_path, "cfg.json")
    cfg = tyro.cli(Config, args=unknown_args, default=Config(**json.load(open(saved_cli_path, "r"))))

    set_seeds()

    if cli.iteration is None:
        load_iteration = searchForMaxIteration(os.path.join(cli.model_path, "point_cloud"))
    else:
        load_iteration = cli.iteration
    print("Loading trained model at iteration {}".format(load_iteration))

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=load_iteration, shuffle=False, model_path=cli.model_path)
    views = scene.getTrainCameras()

    raytracer = GaussianRaytracer(gaussians, views[0].image_width, views[0].image_height)

    # * Run twice to warmup caches
    measure_fps(
        cfg,
        views,
        raytracer,
    )
