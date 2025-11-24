import math
import random

import torch


def transform_normals_to_world(normals_camera, view_matrix):
    # Invert the direction of the camera normals
    normals_camera = -1 * normals_camera

    # Normalize input normals
    normals_camera = normals_camera / torch.norm(normals_camera, dim=-1, keepdim=True)

    # Transform normals from camera to world space
    normals_world = torch.einsum("ij,...j->...i", view_matrix, normals_camera)
    return normals_world


def transform_normals_world_to_camera(normals_world, view_matrix):
    # Transform normals from world to camera space
    normals_camera = torch.einsum("ij,...j->...i", view_matrix.T, normals_world)
    # Invert the direction of the camera normals
    normals_camera = -1 * normals_camera
    return normals_camera


@torch.no_grad()
def compute_primary_ray_directions(
    height: int,
    width: int,
    vertical_fov_radians: float,
    rotation_c2w: torch.Tensor,  # 3x3 matrix, camera → world
) -> torch.Tensor:
    """Compute world-space primary ray directions for every pixel (H, W, 3)."""
    device = rotation_c2w.device
    dtype = rotation_c2w.dtype

    view_size = math.tan(vertical_fov_radians * 0.5)
    aspect = width / float(height)

    # Pixel coordinate grid (centered on pixel centers)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )

    u = (xs + 0.5) / float(width)
    v = (ys + 0.5) / float(height)

    x = aspect * view_size * (2.0 * u - 1.0)
    y = view_size * (1.0 - 2.0 * v)

    # Directions in camera space
    dirs_cam = torch.stack([x, y, -torch.ones_like(x)], dim=-1)  # (H, W, 3)

    # Rotate to world space (c2w)
    dirs_world = dirs_cam @ rotation_c2w.T  # (H, W, 3)

    # Normalize
    dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True)

    return dirs_world


def transform_depth_to_position_image(depth, fov_x_rad, fov_y_rad):
    """
    Convert a depth image to a position image using FoV and image size.

    Args:
        depth (torch.Tensor): Depth image of shape (H, W)
        fov_x_rad (float): Horizontal field of view in radians

    Returns:
        torch.Tensor: Position image of shape (H, W, 3)
    """
    device = depth.device
    H, W = depth.shape

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(fov_x_rad / 2))
    fy = H / (2 * math.tan(fov_y_rad / 2))

    # Principal point at center
    cx = W / 2
    cy = H / 2

    # Create pixel grid
    u = torch.arange(W, device=device).float()
    v = torch.arange(H, device=device).float()
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")

    Z = depth
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy

    position = torch.stack((X, Y, Z), dim=-1)  # Shape: (H, W, 3)
    return position


def transform_points(points: torch.Tensor, transformation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 transformation matrix to 3D points of arbitrary shape ending in 3.

    Parameters:
    - points: torch.Tensor of shape (..., 3)
    - transformation_matrix: torch.Tensor of shape (4, 4)

    Returns:
    - Transformed points: torch.Tensor of shape (..., 3)
    """
    if points.shape[-1] != 3:
        raise ValueError("The last dimension of `points` must be 3")
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be of shape (4, 4)")

    # Flatten to (..., 3)
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, 3)

    # Append homogeneous coordinate
    ones = torch.ones((points_flat.shape[0], 1), dtype=points.dtype, device=points.device)
    points_homogeneous = torch.cat([points_flat, ones], dim=1)  # shape: (-1, 4)

    # Apply transformation
    transformed = points_homogeneous @ transformation_matrix.T  # shape: (-1, 4)

    # Return to original shape with last dimension 3
    return transformed[:, :3].reshape(*original_shape, 3)


def project_pointcloud_to_depth_map(points, fov_x_rad, fov_y_rad, image_size):
    """
    points: (N, 3) 3D point cloud
    fov_x_rad: horizontal FOV in radians
    fov_y_rad: vertical FOV in radians
    image_size: (H, W)

    Returns: (H, W) torch tensor with depth values, 0 for background
    """
    H, W = image_size
    device = points.device

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(fov_x_rad / 2))
    fy = H / (2 * math.tan(fov_y_rad / 2))

    # Principal point at center
    cx = W / 2
    cy = H / 2

    # Unpack and filter
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Project
    u = (x * fx / z + cx).round().long()
    v = (y * fy / z + cy).round().long()

    # Filter in-bounds
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[mask], v[mask], z[mask]
    lin_idx = v * W + u

    # Sort by pixel index and depth
    sorted_idx = lin_idx.argsort()
    lin_idx_sorted = lin_idx[sorted_idx]

    # Keep only nearest (first occurrence per index)
    keep = torch.ones_like(lin_idx_sorted, dtype=torch.bool)
    keep[1:] = lin_idx_sorted[1:] != lin_idx_sorted[:-1]

    u_final = u[sorted_idx][keep]
    v_final = v[sorted_idx][keep]
    z_final = z[sorted_idx][keep]

    # Write depth map
    depth_map = torch.zeros((H, W), device=device)
    depth_map[v_final, u_final] = z_final

    return depth_map


def linear_least_squares_1d(x: torch.Tensor, y: torch.Tensor):
    """
    Fits y ≈ w * x + b using least squares.
    x: (N,) 1D input tensor
    y: (N,) 1D target tensor

    Returns:
        w: scalar weight
        b: scalar bias
    """
    assert x.ndim == 1 and y.ndim == 1
    N = x.shape[0]

    # Design matrix: [x, 1]
    X = torch.stack([x, torch.ones(N, device=x.device)], dim=1)  # (N, 2)

    # Solve least squares: theta = (X^T X)^(-1) X^T y
    theta = torch.linalg.lstsq(X, y).solution  # (2,)

    w, b = theta[0].item(), theta[1].item()
    return w, b


def ransac_linear_fit(x, y, num_iters=100, sample_fraction=0.1, max_sample_size=50, best_fraction=0.1):
    """
    Robustly fits y = wx + b using RANSAC,
    keeping only the best `best_fraction` of points (lowest residuals) for scoring and final fit.
    Automatically chooses sample size for candidate fits.

    Args:
        x (Tensor): shape [N]
        y (Tensor): shape [N]
        num_iters (int): RANSAC iterations
        sample_fraction (float): fraction of points to sample for candidate fits
        max_sample_size (int): max number of points to sample in a single iteration
        best_fraction (float): fraction of points to keep for scoring/refitting (0 < best_fraction <= 1)

    Returns:
        best_model: (w, b)
        best_inliers: boolean tensor of inlier flags
    """
    assert x.shape == y.shape
    assert 0 < best_fraction <= 1, "best_fraction must be between 0 and 1"

    N = x.shape[0]

    # Auto-scale sample size
    sample_size = min(max(2, math.ceil(N * sample_fraction)), max_sample_size)

    top_k = max(1, math.ceil(N * best_fraction))
    best_model = None
    best_inliers = None
    best_error = None

    for _ in range(num_iters):
        # 1. Randomly sample points from all data (classic RANSAC)
        idxs = random.sample(range(N), sample_size)
        x_sample = x[idxs]
        y_sample = y[idxs]

        # Fit y = wx + b
        X_sample = torch.stack([x_sample, torch.ones_like(x_sample)], dim=1)
        w_b = torch.linalg.lstsq(X_sample, y_sample.unsqueeze(1)).solution.squeeze()
        if w_b.ndim == 0:
            continue
        w, b = w_b[0], w_b[1]

        # 2. Compute residuals
        y_pred = w * x + b
        residuals = torch.abs(y - y_pred)

        # 3. Find top `best_fraction` smallest residuals
        _, best_idx = torch.topk(-residuals, top_k)
        inliers = torch.zeros_like(residuals, dtype=torch.bool)
        inliers[best_idx] = True

        # 4. Compute SSE on these inliers
        error = residuals[best_idx].pow(2).sum()

        # 5. Keep best model
        if best_error is None or error < best_error:
            best_model = (w, b)
            best_inliers = inliers
            best_error = error

    # 6. Refit on best `best_fraction` inliers
    if best_model is not None and best_inliers is not None:
        x_in = x[best_inliers]
        y_in = y[best_inliers]
        X_in = torch.stack([x_in, torch.ones_like(x_in)], dim=1)
        w_b = torch.linalg.lstsq(X_in, y_in.unsqueeze(1)).solution.squeeze()
        return (w_b[0], w_b[1]), best_inliers

    return None, None
