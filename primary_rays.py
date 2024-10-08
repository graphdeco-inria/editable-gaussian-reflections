import torch 
from kornia import create_meshgrid

def _get_rays(directions, R_blender, T_blender):
    rays_d = directions @ R_blender.float().T  # (H, W, 3) # !!! review convention
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = T_blender.float().expand(rays_d.shape)  # (H, W, 3) #!!! review convention

    return rays_o, rays_d

def _get_ray_directions(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], -torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def compute_primary_rays(camera_colmap):
    directions = _get_ray_directions(1024//2, 1536//2, [camera_colmap.focal_x/2, -camera_colmap.focal_y/2])  # (h, w, 3) 
    ray_z = torch.norm(directions, dim=-1, keepdim=True) # torch.ones_like(directions[..., :1])  # (h, w, 1)
    directions = directions / ray_z

    R_blender = torch.from_numpy(camera_colmap.R).float()
    T_blender = torch.from_numpy(camera_colmap.T).float()

    rays_o, rays_d = _get_rays(directions, R_blender, T_blender)  
    ray_hits = rays_o + rays_d * ray_z  # (H, W, 3)

    return rays_o.cuda(), rays_d.cuda(), ray_hits.cuda()


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    
    import numpy as np
    class mock_camera:
        R = np.array([[ 8.95075817e-01, -4.21339499e-01,  1.45987016e-01],
            [-4.45913858e-01, -8.45748095e-01,  2.93037404e-01],
            [ 6.64981544e-09, -3.27388423e-01, -9.44889875e-01]])
        T = np.array([3.60848531e-08, 5.08675688e-08, 4.03112897e+00])
        focal_x = 1111.1111350971692
        focal_y = 1111.1111350971692

    prays_o, prays_d = compute_primary_rays(mock_camera)
