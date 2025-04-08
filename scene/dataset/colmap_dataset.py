import os

from arguments import ModelParams

from .camera_info import CameraInfo
from .colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


class ColmapDataset:
    def __init__(
        self,
        model_params: ModelParams,
        data_dir: str,
        split: str = "train",
        images_folder_name: str | None = None,
        llffhold: int = 8,
    ):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split
        self.do_eval = model_params.eval
        self.images_folder_name = (
            images_folder_name if images_folder_name != None else "images"
        )
        self.images_folder = os.path.join(self.data_dir, self.images_folder_name)
        assert model_params.linear_space

        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        keys = sorted(enumerate(cam_extrinsics), key=lambda x: x[0])
        if self.do_eval:
            if split == "train":
                self.keys = [key for idx, key in enumerate(keys) if idx & llffhold != 0]
            else:
                self.keys = [key for idx, key in enumerate(keys) if idx & llffhold == 0]
        else:
            if split == "train":
                self.keys = keys
            else:
                self.keys = []

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> CameraInfo:
        key = self.keys[idx]

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if "LOAD_FROM_IMAGE_FILES" not in os.environ:
            cache_path = (
                image_path.replace("/renders/", "/cache/")
                .replace("render/render_", "")
                .replace(".png", ".pt")
            )
            image_tensor = torch.load(cache_path)
            (
                image,
                diffuse_image,
                glossy_image,
                normal_image,
                position_image,
                roughness_image,
                specular_image,
                metalness_image,
                base_color_image,
                brdf_image,
            ) = torch.unbind(image_tensor, dim=0)
            height, width = image.shape[0], image.shape[1]
        else:
            image = imread(image_path, "render")
            diffuse_image = imread(image_path, "diffuse")
            glossy_image = imread(image_path, "glossy")
            normal_image = imread(image_path, "normal")
            position_image = imread(image_path, "position")
            roughness_image = imread(image_path, "roughness")
            specular_image = imread(image_path, "specular")
            metalness_image = imread(image_path, "metalness")
            base_color_image = imread(image_path, "base_color")
            brdf_image = imread(image_path, "glossy_brdf")
            height, width = image.shape[0], image.shape[1]
        diffuse_image = diffuse_image * self.model_params.exposure
        glossy_image = glossy_image * self.model_params.exposure

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            diffuse_image=diffuse_image,
            glossy_image=glossy_image,
            position_image=position_image,
            normal_image=normal_image,
            roughness_image=roughness_image,
            metalness_image=metalness_image,
            base_color_image=base_color_image,
            brdf_image=brdf_image,
            specular_image=specular_image,
        )
        return cam_info
