import numpy as np
import json 
import os
import math 

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def colamp_pose_to_nerf_pose(qvec, tvec):  # Taken from instantngp
    qvec = np.copy(qvec)
    tvec = np.copy(tvec)
    def qvec2rotmat(qvec):
        return np.array(
            [
                [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
                [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
            ]
        )

    R = qvec2rotmat(-qvec)
    t = tvec.reshape([3, 1])
    m = np.concatenate([np.concatenate([R, t], 1), np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])], 0)
    c2w = np.matmul(m, flip_mat)

    return c2w

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
               
def nerf_pose_to_colmap_pose(c2w):  # Written by inverting each line 1 by 1
    c2w = np.matmul(c2w, np.linalg.inv(flip_mat))
    m = np.linalg.inv(c2w)
    R, tvec = m[:3, :3], m[:3, -1]
    qvec = rotmat2qvec(R)
    return qvec, tvec

def nerf_to_colmap(json_path):
    outdir = os.path.dirname(os.path.realpath(json_path)) + "/fake_sparse"
    os.makedirs(outdir, exist_ok=True)

    transforms = json.load(open(json_path, 'r'))
    w = transforms["width"]
    h = transforms["height"]

    with open(outdir + "/points3D.txt", "w") as file:
        pass

    with open(outdir + "/cameras.txt", "w") as file:
        focal = round(w / (2 * math.tan(transforms["camera_angle_x"] / 2)), 2)
        print(f"""# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE {w} {h} {round(focal, 2)} {round(focal, 2)} {int(w/2)} {int(h/2)}
""".strip(), file=file, end="")
        
    with open(outdir + "/images.txt", "w") as file:
        print("""# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 200, mean observations per image: 2233.04
""".strip(), file=file)
        
        for i, frame in enumerate(transforms["frames"]):
            id = int(frame["file_path"].split("/")[-1].split(".")[0].split("_")[1])
            print(id + 1, file=file, end=" ")
            c2w = np.array(frame["transform_matrix"])
            qvec, tvec = nerf_pose_to_colmap_pose(c2w)
            for x in qvec:
                print(x, file=file, end=" ")
            for x in tvec:
                print(x, file=file, end=" ")
            print(1, file=file, end=" ")

            print(frame["file_path"].split("/")[-1] + ".png", file=file, end=" ")

            print(file=file) # Need a blank line where points will be
            print(file=file)


if __name__ == "__main__":
    import sys 
    nerf_to_colmap(sys.argv[1])
