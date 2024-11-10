import math 
from PIL import Image 
from torchvision.transforms.functional import to_tensor
import torch 
from torchvision.utils import save_image
import kornia
import torch.nn.functional as F
import os 
from kornia.color.lab import rgb_to_lab, lab_to_rgb


# ----------------------------------------

def dot(a, b, dim=-1):
    return (a * b).sum(dim=dim, keepdim=True)

def dotpos(a, b, dim=-1):
    return (a * b).sum(dim=dim, keepdim=True).clamp(0.0)

def normalize(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)

# ----------------------------------------

def cook_torrence(normals, view_dir, light_dir, base_color, metalness, roughness):
    halfway = normalize(view_dir + light_dir)

    diffuse = dotpos(normals, light_dir) / math.pi
    
    D = D_ggx(normals, halfway, roughness)
    G = G_ggx(normals, view_dir, light_dir, base_color, roughness)
    specular = D * G / (4.0 * dotpos(normals, view_dir) * dotpos(normals, light_dir) + 1e-8)
    
    k = F_schlick(normals, halfway, base_color, metalness).clamp(0.0)
    return (1.0 - k) * diffuse * base_color + k * specular

def cook_torrence_specular(normals, view_dir, light_dir, metalness, roughness):
    halfway = normalize(view_dir + light_dir)
    
    D = D_ggx(normals, halfway, roughness)
    G = G_ggx(normals, view_dir, light_dir, roughness)
    specular = D * G / (4.0 * dotpos(normals, view_dir) * dotpos(normals, light_dir) + 1e-8)
    
    return specular #F_schlick_specular(normals, halfway, metalness).clamp(0.0) * 

def F_schlick_specular(normals, halfway, metalness, base_reflectivity=0.04):
    r0 = (1.0 - metalness) * base_reflectivity + metalness 
    return r0 + (1.0 - r0) * (1.0 - dotpos(normals, halfway))**5

def F_schlick(normals, halfway, base_color, metalness, base_reflectivity=0.04):
    r0 = (1.0 - metalness) * base_reflectivity + metalness * base_color 
    return r0 + (1.0 - r0) * (1.0 - dotpos(normals, halfway))**5

def D_ggx(normals, halfway, roughness):
    alpha = roughness**2
    return alpha**2 / (math.pi * (dotpos(normals, halfway)**2*(alpha**2 - 1) + 1))**2

def G_ggx(normals, view_dir, light_dir, roughness):
    k_direct = (roughness**2 + 1)**2 / 8 
    G_v = dotpos(normals, view_dir) / (dotpos(normals, view_dir)*(1.0 - k_direct) + k_direct)
    G_l = dotpos(normals, light_dir) / (dotpos(normals, light_dir)*(1.0 - k_direct) + k_direct)
    return G_v * G_l

# ---------------------------------------

RESOLUTION = 512
camera_angle_x = 0.6733496189117432
focal = 0.5 * RESOLUTION / torch.tan(torch.tensor(0.5) * camera_angle_x)

def project(ws_pts, view_idx, poses):
    c2w = poses[view_idx].to(ws_pts.device)
    centered_pts = ws_pts.unsqueeze(-2) - c2w[:, :3, 3]
    cs_pts = (centered_pts.unsqueeze(-2) @ c2w[:, :3, :3]).squeeze(-2) #?? squeeze on 2 or 1
    ss_pts = cs_pts[..., :2] / cs_pts[..., 2:3] * focal / (RESOLUTION / 2)
    return ss_pts.moveaxis(-2, 0)
