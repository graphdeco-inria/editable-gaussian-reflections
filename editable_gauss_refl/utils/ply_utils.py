import numpy as np
import plyfile


def read_ply(ply_path: str):
    # Read the .ply file
    plydata = plyfile.PlyData.read(ply_path)
    vertex_data = plydata["vertex"].data
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    colors = np.vstack([vertex_data["red"], vertex_data["green"], vertex_data["blue"]]).T
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    return points, colors


def save_ply(ply_path: str, points: np.ndarray, colors: np.ndarray):
    # Create structured array
    vertex = np.array(
        [(*point, *color) for point, color in zip(points, colors)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
        ],
    )

    # Create a PlyElement
    ply_element = plyfile.PlyElement.describe(vertex, "vertex")

    # Write to PLY file
    plyfile.PlyData([ply_element], text=True).write(ply_path)
