import os
from tqdm import tqdm
import torch
import numpy as np

from gaussian_tracing.dataset import BlenderDataset
from gaussian_tracing.dataset.image_utils import to_pil_image
from gaussian_tracing.utils.tonemapping import tonemap
from gaussian_tracing.utils.depth_utils import transform_normals_world_to_camera
from gaussian_tracing.utils.depth_utils import transform_distance_to_position_image

def main():
    scene_names = [
        "shiny_kitchen", 
        "shiny_bedroom", 
        "shiny_livingroom", 
        "shiny_office",
        "multichromeball_identical_kitchen_v2", 
        "multichromeball_kitchen_v2", 
        "multichromeball_tint_kitchen_v2", 
        "multichromeball_value_kitchen_v2",
    ]
    split_names = [
        "test",
        "train",
    ]
    buffer_names = [
        "render",
        "albedo",
        "diffuse",
        "glossy",
        "roughness",
        "metalness",
        "depth",
        "normal",
        "base_color",
        "specular",
        "brdf",
    ]

    dataset_dir = "./data/renders"
    out_dir = f"./output/renders_compressed"

    for scene_name in scene_names:
        data_dir = os.path.join(dataset_dir, scene_name)
        for split_name in split_names:
            dataset = BlenderDataset(
                data_dir=data_dir,
                split=split_name,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=lambda x: x,
                persistent_workers=True,
            )
            for cam_info_batch in tqdm(dataloader):
                cam_info = cam_info_batch[0]
                frame_id = os.path.splitext(cam_info.image_name)[0].split("_")[-1]
                for buffer_name in buffer_names:
                    buffer = getattr(cam_info, f"{buffer_name}_image") if buffer_name != "render" else cam_info.image
                    buffer_ext = ".png"
                    buffer_path = os.path.join(out_dir, scene_name, split_name, buffer_name, f"{buffer_name}_{frame_id}{buffer_ext}")
                    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

                    if buffer_name in ["render", "diffuse", "glossy"]:
                        buffer = tonemap(buffer * 3.5)
                    elif buffer_name in ["albedo", "base_color", "brdf"]:
                        buffer = buffer.clip(0.0, 1.0)
                    elif buffer_name in ["roughness", "metalness", "specular"]:
                        buffer = buffer[:,:,:1]
                    elif buffer_name == "depth":
                        distance_image = buffer[:,:,None]
                        position_image = transform_distance_to_position_image(
                            distance_image[:,:,0], cam_info.FovX, cam_info.FovY
                        )
                        depth_image = position_image[:,:,2]
                        buffer = depth_image[:,:,None]

                        lower = 0.0 # buffer.min()
                        upper = 4.0 # buffer.max()
                        buffer = (buffer - lower) / (upper - lower)
                    elif buffer_name == "normal":
                        buffer = transform_normals_world_to_camera(buffer, torch.tensor(cam_info.R, dtype=buffer.dtype))
                        buffer = buffer * 0.5 + 0.5
                    else:
                        raise ValueError(f"Buffer name not recognized: {buffer_name}")

                    if buffer_ext == ".png":
                        buffer_pil = to_pil_image(buffer.numpy())
                        buffer_pil.save(buffer_path)
                    else:
                        np.save(buffer_path, buffer.numpy())

if __name__ == "__main__":
    main()
