import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from utils.general_utils import build_rotation
from dataclasses import dataclass

try:
    from bvh_tracing import _C
except Exception as e:
    _src_path = os.path.dirname(os.path.abspath(__file__))
    _C = load(
        name='_bvh_tracing',
        extra_cuda_cflags=["-O3", "--expt-extended-lambda"],
        extra_cflags=["-O3"],
        sources=[os.path.join(_src_path, 'src', f) for f in [
            'bvh.cu',
            'trace.cu',
            'construct.cu',
            'bindings.cpp',
        ]],
        extra_include_paths=[
            os.path.join(_src_path, 'include'),
        ],
        verbose=True)


class RayTracer:
    def __init__(self, means3D, scales, rotations):
        P = means3D.shape[0]
        rot = build_rotation(rotations)
        nodes = torch.full((2 * P - 1, 5), -1, device="cuda").int()
        nodes[:P - 1, 4] = 0
        nodes[P - 1:, 4] = 1
        aabbs = torch.zeros(2 * P - 1, 6, device="cuda").float()
        aabbs[:, :3] = 100000
        aabbs[:, 3:] = -100000

        a, b, c = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
        m = 3
        sa, sb, sc = m * scales[:, 0], m * scales[:, 1], m * scales[:, 2]

        x111 = means3D + a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x110 = means3D + a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x101 = means3D + a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x100 = means3D + a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        x011 = means3D - a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x010 = means3D - a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x001 = means3D - a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x000 = means3D - a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        aabb_min = torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(torch.minimum(torch.minimum(torch.minimum(x111, x110), x101), x100), x011),
                          x010), x001), x000)
        aabb_max = torch.maximum(torch.maximum(torch.maximum(torch.maximum(
            torch.maximum(torch.maximum(torch.maximum(x111, x110), x101), x100), x011), x010), x001), x000)

        aabbs[P - 1:] = torch.cat([aabb_min, aabb_max], dim=-1)

        self.tree, self.aabb, self.morton = _C.create_bvh(means3D, scales, rotations, nodes, aabbs)

    @torch.no_grad()
    def trace_visibility(self, rays_o, rays_d, means3D, symm_inv, opacity, normals):
        cotrib, opa = _C.trace_bvh_opacity(self.tree, self.aabb,
                                                 rays_o, rays_d,
                                                 means3D, symm_inv,
                                                 opacity, normals)
        return {
            "visibility": opa.unsqueeze(-1),
            "contribute": cotrib.unsqueeze(-1),
        }
    
    @torch.no_grad()
    def trace(self, rays_o, rays_d, means3D, symm_inv, opacity):
        num_rendered, point_list_vec, position_list_vec, ray_id_list_vec = _C.trace_bvh(self.tree, self.aabb,
                                                 rays_o, rays_d,
                                                 means3D, symm_inv,
                                                 opacity)
        olist = ray_id_list_vec.squeeze(-1)
        dlist = point_list_vec.squeeze(-1)
        return Intersections(olist[dlist != -1], dlist[dlist != -1], position_list_vec[dlist != -1])


@dataclass 
class Intersections:
    ray_ids: torch.Tensor
    "torch.Size([total_num_intersections]); these are sorted e.g. it will look like [0, 0, 1, 1, 1, 1, 2, 2, ...]."

    gaussian_ids: torch.Tensor
    "torch.Size([total_num_intersections]); for each ray_id, these are sorted by distance, from back to front"
    
    hit_positions: torch.Tensor
    "torch.Size([total_num_intersections, 3]); for each ray_id, these are sorted by distance, from back to front"

    def __post_init__(self):
        assert len(self.ray_ids) == len(self.gaussian_ids) == len(self.hit_positions)
        
    def __len__(self):
        return len(self.ray_ids)

    @staticmethod
    def combine(all_hits):
        return Intersections(
            ray_ids=torch.cat([hits.ray_ids for hits in all_hits]),
            gaussian_ids=torch.cat([hits.gaussian_ids for hits in all_hits]),
            hit_positions=torch.cat([hits.hit_positions for hits in all_hits]),
        )