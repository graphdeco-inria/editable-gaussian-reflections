#
# MIT License
#
# Copyright (c) 2024 3D Vision Group at the State Key Lab of CAD&CG,
# Zhejiang University.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copied from here:
#   https://github.com/zju3dv/EnvGS/blob/main/easyvolcap/utils/cam_utils.py


import numpy as np

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-13)


def compute_center_of_attention(c2ws: np.ndarray):
    # TODO: Should vectorize this to make it faster, this is not very tom94
    totw = 0.0
    totp = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    for mf in c2ws:
        for mg in c2ws:
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    return totp[..., None]  # 3, 1


def closest_point_2_lines(oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray):
    # Returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def average_c2ws(c2ws: np.ndarray, align_cameras: bool = True, look_at_center: bool = True) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """

    if align_cameras:
        # 1. Compute the center
        center = compute_center_of_attention(c2ws)[..., 0]  # (3)
        # 2. Compute the z axis
        z = -normalize(c2ws[..., 1].mean(0))  # (3) # FIXME: WHY?
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 2].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    else:
        # 1. Compute the center
        center = c2ws[..., 3].mean(0)  # (3)
        # 2. Compute the z axis
        if look_at_center:
            look = compute_center_of_attention(c2ws)[..., 0]  # (3)
            z = normalize(look - center)
        else:
            z = normalize(c2ws[..., 2].mean(0))  # (3)
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 1].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    c2w_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return c2w_avg


def generate_spiral_path(
    c2ws: np.ndarray,
    n_render_views=300,
    n_rots=2,
    zrate=0.5,
    percentile=70,
    focal_offset=0.0,
    radius_ratio=1.0,
    xyz_ratio=[1.0, 1.0, 0.25],
    xyz_offset=[0.0, 0.0, 0.0],
    **kwargs,
) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.
    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py
    """
    # Prepare input data
    c2ws = c2ws[..., :3, :4]

    # Center pose
    c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)  # [3, 4]

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    focal = focal_offset + np.linalg.norm(compute_center_of_attention(c2ws)[..., 0] - c2w_avg[..., 3])  # (3)

    # Get radii for spiral path using 70th percentile of camera origins.
    radii = np.percentile(np.abs(c2ws[:, :3, 3] - c2w_avg[..., 3]), percentile, 0) * radius_ratio  # N, 3
    radii = np.concatenate([xyz_ratio * radii, [1.0]])  # 4,

    # Generate c2ws for spiral path.
    render_c2ws = []
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_render_views, endpoint=False):
        t = radii * [
            np.cos(theta),
            np.sin(theta),
            np.sin(theta * zrate),
            1.0,
        ] + np.concatenate([xyz_offset, [0.0]])

        center = c2w_avg @ t
        center = center.astype(c2ws.dtype)
        lookat = c2w_avg @ np.array([0, 0, focal, 1.0], dtype=c2ws.dtype)

        v_front = -normalize(center - lookat)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws
