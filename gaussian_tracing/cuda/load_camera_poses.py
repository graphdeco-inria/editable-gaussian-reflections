import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

SCENE_PATH = "data/hard_kitchen_mirror"  #!! was data/shiny_kitchen
DOWNSAMPLING = 2

IMAGE_HEIGHT = 1024 // DOWNSAMPLING
IMAGE_WIDTH = 1536 // DOWNSAMPLING

NUM_TRAIN_IMAGES = 200  #!!
NUM_TEST_IMAGES = 100  #!!

train_cam_poses = [
    torch.tensor(x["transform_matrix"])
    for x in sorted(
        json.load(open(f"{SCENE_PATH}/transforms_train.json", "rb"))["frames"],
        key=lambda frame: frame["file_path"],
    )
][:NUM_TRAIN_IMAGES]
test_cam_poses = [
    torch.tensor(x["transform_matrix"])
    for x in sorted(
        json.load(open(f"{SCENE_PATH}/transforms_test.json", "rb"))["frames"],
        key=lambda frame: frame["file_path"],
    )
][:NUM_TEST_IMAGES]


def load_image(split, i):
    x = (
        TF.to_tensor(
            Image.open(f"{SCENE_PATH}/{split}/render/render_{i:04d}.png").convert("RGB")
        )
        .cuda()
        .moveaxis(0, -1)
        .contiguous()
    )
    return (
        F.avg_pool2d(x.moveaxis(-1, 0)[None], DOWNSAMPLING).float()[0].moveaxis(0, -1)
    )


def load_image_exr(split, i, label="render"):
    x = torch.tensor(
        cv2.imread(
            f"{SCENE_PATH}/{split}/{label}/{label}_{i:04d}.exr", cv2.IMREAD_UNCHANGED
        ),
        device="cuda",
    ).contiguous()
    return (
        F.avg_pool2d(x.moveaxis(-1, 0)[None], DOWNSAMPLING).float()[0].moveaxis(0, -1)
    )


with ThreadPoolExecutor(max_workers=4) as executor:
    train_images = list(
        tqdm(
            executor.map(lambda i: load_image("train", i), range(NUM_TRAIN_IMAGES)),
            total=NUM_TRAIN_IMAGES,
            desc="Loading train data",
        )
    )
    test_images = list(
        tqdm(
            executor.map(lambda i: load_image("test", i), range(NUM_TEST_IMAGES)),
            total=NUM_TEST_IMAGES,
            desc="Loading test data",
        )
    )

    apply = executor.map
    if "LOAD_ALL_MAPS" in os.environ:
        train_diffuse_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "diffuse"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train diffuse maps",
            )
        )
        train_glossy_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "glossy"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train glossy maps",
            )
        )
        train_normal_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "normal"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train normal maps",
            )
        )
        train_roughness_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "roughness").mean(
                        dim=-1, keepdim=True
                    ),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train roughness maps",
            )
        )
        train_specular_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "specular"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train specular maps",
            )
        )
        train_metalness_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "metalness"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train metalness maps",
            )
        )
        train_brdf_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "glossy_brdf"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train glossy BRDF maps",
            )
        )
        train_base_color_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("train", i, "base_color"),
                    range(NUM_TRAIN_IMAGES),
                ),
                total=NUM_TRAIN_IMAGES,
                desc="Loading train base color maps",
            )
        )
        train_f0_images = [
            (1.0 - train_metalness_images[i]) * 0.08 * train_specular_images[i]
            + train_metalness_images[i] * train_base_color_images[i]
            for i in range(NUM_TRAIN_IMAGES)
        ]

        test_diffuse_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "diffuse"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test diffuse maps",
            )
        )
        test_glossy_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "glossy"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test glossy maps",
            )
        )
        test_normal_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "normal"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test normal maps",
            )
        )
        test_roughness_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "roughness").mean(
                        dim=-1, keepdim=True
                    ),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test roughness maps",
            )
        )
        test_specular_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "specular"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test specular maps",
            )
        )
        test_metalness_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "metalness"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test metalness maps",
            )
        )
        test_brdf_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "glossy_brdf"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test glossy BRDF maps",
            )
        )
        test_base_color_images = list(
            tqdm(
                apply(
                    lambda i: load_image_exr("test", i, "base_color"),
                    range(NUM_TEST_IMAGES),
                ),
                total=NUM_TEST_IMAGES,
                desc="Loading test base color maps",
            )
        )
        test_f0_images = [
            (1.0 - test_metalness_images[i]) * 0.08 * test_specular_images[i]
            + test_metalness_images[i] * test_base_color_images[i]
            for i in range(NUM_TEST_IMAGES)
        ]


# concat all train images (reegular + diffuse + glossy + ...) channelwise then dump them as train_images.pt
# train_images = torch.cat([train_images]
