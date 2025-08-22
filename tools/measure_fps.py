import os
import tyro
import torch
from tqdm import tqdm

from gaussian_tracing.arguments import (
    TyroConfig,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.general_utils import safe_state


@torch.no_grad()
def render_set(
    model_path,
    views,
    raytracer,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for view in tqdm(views, desc="Rendering progress"):
        _ = render(view, raytracer, force_update_bvh=False, targets_available=False)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_sec = elapsed_ms / 1000.0
    fps = len(views) / elapsed_sec
    print(f"{fps:.2f} FPS")

    with open(os.path.join(model_path, "fps.txt"), "w") as f:
        f.write(f"{fps:.2f}\n")


def main(cfg: TyroConfig):
    # Initialize system state (RNG)
    safe_state(cfg.quiet)

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=cfg.iteration, shuffle=False)
    views = scene.getTrainCameras()

    raytracer = GaussianRaytracer(
        gaussians, views[0].image_width, views[0].image_height
    )

    # Run twice. First run is always slow.
    for _ in range(2):
        render_set(
            cfg.model_path,
            views,
            raytracer,
        )

if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)
    main(cfg)
