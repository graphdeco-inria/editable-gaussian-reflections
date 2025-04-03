import torch
from bvh import RayTracer
from gauss_render import compute_gaussian_alphas
from load_model import *
from primary_rays import compute_primary_rays
from torchvision.utils import save_image
import nerfacc
import matplotlib.pyplot as plt
import sys
import os
import math
from scene import GaussianModel
from gaussian_renderer import render
import torch.nn.functional as F


pc = GaussianModel(None)
pc.load_ply("hard_kitchen_7k_pc.ply")

save_image(
    F.avg_pool2d(render(camera_colmap, pc, args)["render"][None], 2), "render_splat.png"
)

with torch.no_grad():
    pc._xyz.nan_to_num_(0.0, 0.0, 0.0)
with torch.no_grad():
    pc._scaling.nan_to_num_(0.0, 0.0, 0.0)
with torch.no_grad():
    pc._rotation.nan_to_num_(0.0, 0.0, 0.0)
with torch.no_grad():
    pc._features_dc.nan_to_num_(0.0, 0.0, 0.0)

# --------------------------

ray_o, ray_d, ray_imageplane_hitpoints_ws = compute_primary_rays(camera)

if False:
    import numpy as np
    from plyfile import PlyData, PlyElement

    vertices = np.array(ray_imageplane_hitpoints_ws.flatten(0, 1).cpu())
    el = PlyElement.describe(
        np.array(
            [tuple(v) for v in vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        ),
        "vertex",
    )
    PlyData([el]).write("ray_imageplane_hitpoints_ws.ply")

    import numpy as np
    from plyfile import PlyData, PlyElement

    vertices = np.array((ray_o + ray_d).cpu().flatten(0, 1))
    el = PlyElement.describe(
        np.array(
            [tuple(v) for v in vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        ),
        "vertex",
    )
    PlyData([el]).write("rays.ply")

    import numpy as np
    from plyfile import PlyData, PlyElement

    vertices = np.array((ray_o + torch.randn_like(ray_o)).cpu().flatten(0, 1))
    el = PlyElement.describe(
        np.array(
            [tuple(v) for v in vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        ),
        "vertex",
    )
    PlyData([el]).write("origins.ply")

print("-> done gen rays", flush=True)

# --------------------------

import time

times = open("times.txt", "w")
devnull = open(os.devnull, "w")
print("start")
# with torch.no_grad():
#     pc._scaling -= math.log(3)
#     pc._xyz *= 1
raytracer = RayTracer(pc.get_xyz, pc.get_scaling, pc.get_rotation)
start = time.time()
hits = raytracer.trace(
    ray_o.flatten(0, 1),
    ray_d.flatten(0, 1),
    pc.get_xyz,
    pc.get_inverse_covariance(),
    pc.get_opacity,
)
print("--------------->", time.time() - start)

print("-> done tracing", flush=True)

if False:
    buffer = torch.zeros(800 * 800, 3)
    buffer[hits.ray_ids] = hits.hit_positions.cpu()
    save_image(buffer.reshape(800, 800, 3).moveaxis(-1, 0) / 2 + 0.5, "hits.png")

# --------------------------

dx = hits.hit_positions - pc.get_xyz[hits.gaussian_ids]


from gauss_render import *

if False:
    alphas = torch.exp(-((dx).norm(dim=-1) ** 2) * 1000.0) * pc.get_opacity[
        hits.gaussian_ids
    ].squeeze(-1)
else:
    cov2d = build_covariance_2d(
        mean3d=pc.get_xyz,
        cov3d=build_covariance_3d(pc.get_scaling, pc.get_rotation),
        viewmatrix=camera_colmap.world_view_transform,
        fov_x=camera_colmap.FoVx,
        fov_y=camera_colmap.FoVy,
        focal_x=camera_colmap.focal_x / 2,
        focal_y=camera_colmap.focal_y / 2,
    )

    def geom_transform_points(points, transf_matrix):
        """Homogeneous transformation of points.

        Args:
            points: [P, 3]
            transf_matrix: [4, 4]

        Returns:
            [P, 3]
        """
        P, _ = points.shape
        ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
        points_hom = torch.cat([points, ones], dim=1)
        points_out = (transf_matrix @ points_hom[..., None]).squeeze(-1)

        denom = points_out[..., 3:] + 0.0000001
        return (points_out[..., :3] / denom).squeeze(dim=0)

    def computeJacobian(means, camera):
        # Transform points to camera space
        t = geom_transform_points(means, camera.world_view_transform)
        l = t.norm(dim=1, keepdim=True).flatten()

        # Compute the jacobian according to (29) from EWA Volume Splatting M.Zwicker et. al (2001)
        jacobian = torch.zeros(t.shape[0], 3, 3, device=means.device)
        jacobian[:, 0, 0] = 1 / t[:, 2]
        jacobian[:, 0, 2] = -t[:, 0] / t[:, 2] ** 2
        jacobian[:, 1, 1] = 1 / t[:, 2]
        jacobian[:, 1, 2] = -t[:, 1] / t[:, 2] ** 2
        jacobian[:, 2, 0] = t[:, 0] / l
        jacobian[:, 2, 1] = t[:, 1] / l
        jacobian[:, 2, 2] = t[:, 2] / l

        return jacobian

    def covariance_from_3d_to_2d(camera, means, cov3d):
        # h_x = camera.width / (2.0 * math.tan(camera.fovx / 2.0))
        # h_y = camera.height / (2.0 * math.tan(camera.fovy / 2.0))
        h_x = 1.0
        h_y = 1.0

        R = camera.world_view_transform[:3, :3][None, ...]

        J = computeJacobian(means, camera)
        J[:, 0] = J[:, 0] * h_x
        J[:, 1] = J[:, 1] * h_y

        cov = (
            J[None, ...]
            @ R[None, ...]
            @ cov3d
            @ R.transpose(1, 2)[None, ...]
            @ J.transpose(1, 2)[None, ...]
        )  # + torch.eye(3).cuda() * 0.3

        return cov[0, :, :2, :2]

    # cov2d = covariance_from_3d_to_2d(camera_colmap, pc.get_xyz, pc.get_full_covariance())
    # cov2d_inv = cov2d.inverse()[hits.gaussian_ids]
    # alpha = torch.exp(-(1/2.0)*(dx @ cov2d_inv @ dx)) * pc.get_opacity[hits.gaussian_ids].squeeze(-1)

    sorted_conic = cov2d.inverse()[hits.gaussian_ids]

    dx = dx @ torch.tensor(R).cuda().T.float()
    dx = dx * torch.tensor([1024 // 2, 1536 // 2, 1]).cuda()

    gauss_weight = torch.exp(
        -0.5
        * (
            dx[:, 0] ** 2 * sorted_conic[:, 0, 0]
            + dx[:, 1] ** 2 * sorted_conic[:, 1, 1]
            + dx[:, 0] * dx[:, 1] * sorted_conic[:, 0, 1]
            + dx[:, 0] * dx[:, 1] * sorted_conic[:, 1, 0]
        )
    )
    alphas = gauss_weight * pc.get_opacity[hits.gaussian_ids].squeeze(-1)

from utils.sh_utils import eval_sh

shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
dir_pp = pc.get_xyz - camera_colmap.camera_center.repeat(pc.get_features.shape[0], 1)
dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
colors = torch.clamp_min(sh2rgb + 0.5, 0.0)[hits.gaussian_ids]

if False:
    buffer = torch.zeros(800 * 800)
    buffer[hits.ray_ids] = alphas.cpu() * 100
    save_image(buffer.reshape(800, 800, 1).moveaxis(-1, 0).clamp(0, 1), "alphas.png")
    # buffer = torch.zeros(800*800)
    # buffer[hits.ray_ids] = gauss_weight.cpu()*10
    # save_image(buffer.reshape(800, 800, 1).moveaxis(-1, 0), "gauss_weight.png")

weights, transmittances = nerfacc.render_weight_from_alpha(
    alphas=alphas, ray_indices=hits.ray_ids.long()
)

print("-> done compute weights", flush=True)

#  --------------------------

pixel_colors = nerfacc.accumulate_along_rays(
    weights=weights,
    values=colors,
    ray_indices=hits.ray_ids.long(),
    n_rays=ray_o.shape[0] * ray_o.shape[1],
)

print("-> done", flush=True)


print("--------------->", time.time() - start)

# --------------------------
save_image(
    pixel_colors.reshape(1024 // 2, 1536 // 2, 3).moveaxis(-1, 0), "render_raycast.png"
)
