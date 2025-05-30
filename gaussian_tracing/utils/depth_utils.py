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


def transform_distance_to_position_image(distance, fov_x_rad, fov_y_rad):
    """
    Convert a distance image (camera-center to surface point) to a position image.

    Args:
        distance (torch.Tensor): Distance image of shape (H, W)
        fov_x_rad (float): Horizontal field of view in radians
        fov_y_rad (float): Vertical field of view in radians

    Returns:
        torch.Tensor: Position image of shape (H, W, 3)
    """
    device = distance.device
    H, W = distance.shape

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(fov_x_rad / 2))
    fy = H / (2 * math.tan(fov_y_rad / 2))
    cx = W / 2
    cy = H / 2

    # Pixel grid
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")

    # Ray directions in camera space
    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy
    z = torch.ones_like(x)

    # Normalize ray directions
    dirs = torch.stack((x, y, z), dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Scale ray directions by distance
    position = dirs * distance.unsqueeze(-1)
    return position


def transform_points(
    points: torch.Tensor, transformation_matrix: torch.Tensor
) -> torch.Tensor:
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
    ones = torch.ones(
        (points_flat.shape[0], 1), dtype=points.dtype, device=points.device
    )
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


def ransac_linear_fit(x, y, num_iters=100, threshold=None, min_inliers=0.5):
    """
    Robustly fits y = wx + b using RANSAC.

    Args:
        x (Tensor): shape [N]
        y (Tensor): shape [N]
        num_iters (int): RANSAC iterations
        threshold (float): max residual to count as inlier
        min_inliers (float): min fraction of inliers to accept a model

    Returns:
        best_model: (w, b)
        best_inliers: boolean tensor of inlier flags
    """
    assert x.shape == y.shape
    if threshold is None:
        # Initial fit (least squares)
        X = torch.stack([x, torch.ones_like(x)], dim=1)
        w_b = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
        residuals = torch.abs(y - (w_b[0] * x + w_b[1]))

        # Estimate noise level
        sigma = residuals.std()
        threshold = 2.5 * sigma

    N = x.shape[0]
    best_model = None
    best_inliers = None
    best_inlier_count = 0

    for _ in range(num_iters):
        # 1. Sample 2 random points
        idxs = random.sample(range(N), 2)
        x_sample = x[idxs]
        y_sample = y[idxs]

        # 2. Prepare the design matrix
        X_sample = torch.stack([x_sample, torch.ones_like(x_sample)], dim=1)
        y_sample = y_sample.unsqueeze(1)

        # 3. Fit the model using torch.linalg.lstsq
        result = torch.linalg.lstsq(X_sample, y_sample)
        w_b = result.solution.squeeze()
        if w_b.ndim == 0:  # just in case
            continue
        w, b = w_b[0], w_b[1]

        # 4. Compute residuals on all data
        y_pred = w * x + b
        residuals = torch.abs(y - y_pred)

        # 5. Determine inliers
        inliers = residuals < threshold
        inlier_count = inliers.sum().item()

        # 6. Keep the best model
        if inlier_count > best_inlier_count and inlier_count > min_inliers * N:
            best_model = (w, b)
            best_inliers = inliers
            best_inlier_count = inlier_count

    # Optional: Refit on inliers
    if best_model is not None and best_inliers is not None:
        x_in = x[best_inliers]
        y_in = y[best_inliers]
        X_in = torch.stack([x_in, torch.ones_like(x_in)], dim=1)
        y_in = y_in.unsqueeze(1)
        result = torch.linalg.lstsq(X_in, y_in)
        w_b = result.solution.squeeze()
        return (w_b[0], w_b[1]), best_inliers

    return None, None  # No model found
