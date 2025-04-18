# *** conda activate diffusers3

import glob
import random
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.io
import time
from dataclasses import dataclass
import tyro
import tqdm 
from typing import Optional, Literal
import torchvision.transforms.functional as TF 
import torch.nn as nn 
import math 
from utils.image_utils import psnr

@dataclass
class Config:
    lr: float = 2e-4
    num_epochs: int = 100 
    model_path: str = "output/shiny_kitchen_v68"
    iteration: int = 30000
    max_images: Optional[int] = None
    kernel_size: int = 1
    pe: int = 6
    width: int = 256

DOWNSAMPLING = 1

conf = tyro.cli(Config)


# todo include viewing direction and reflection ray direction 
# todo include 
# todo add a psnr eval 
# todo make video render work again
# todo find a faster network that still works (maybe just some conv layers?)
# todo concat other maps to help out
# todo sanity check train on the maps as input, is it just memorizing the data?

# ------------------- Define model


def positional_encoding(positions, freqs=conf.pe): # taken from tensorf
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(conf.width, kernel_size=conf.kernel_size, padding="same"), 
            nn.ReLU(),
            nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(conf.width, 3, kernel_size=conf.kernel_size, padding="same"),
        ).cuda()

    def forward(self, gbuffer):
        inputs = torch.cat([
            gbuffer["render"],
            gbuffer["diffuse"],
            gbuffer["glossy"],
            gbuffer["normal"],
            gbuffer["F0"],
            # gbuffer["position"],
            gbuffer["roughness"],
        ], dim=1)
        inputs_pe = positional_encoding(inputs.moveaxis(1, -1)).moveaxis(-1, 1)
        return self.layers(inputs_pe)

model = Model()    

model.train()
optimizer = optim.Adam(model.parameters(), lr=conf.lr) 
criterion = torch.nn.MSELoss()
transform = transforms.Compose(
    [
        transforms.Resize(
            (512 // DOWNSAMPLING, 768 // DOWNSAMPLING),
            antialias=True,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
)

# -------------------- Load training data

print(f"{conf.model_path}/train/ours_{conf.iteration}")


def open_image(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).cuda()[None]

def open_gbuffers(render_path):
    return {
        "render": open_image(render_path),
        "gt": open_image(render_path.replace("/render/", "/render_gt/")),
        "diffuse": open_image(render_path.replace("/render", "/diffuse").replace("_render", "_diffuse")),
        "glossy": open_image(render_path.replace("/render", "/glossy").replace("_render", "_glossy")),
        "position": open_image(render_path.replace("/render", "/position").replace("_render", "_position")),
        "normal": open_image(render_path.replace("/render", "/normal").replace("_render", "_normal")),
        "roughness": open_image(render_path.replace("/render", "/roughness").replace("_render", "_roughness")),
        "F0": open_image(render_path.replace("/render", "/F0").replace("_render", "_F0")),
    }

train_render_paths = glob.glob(f"{conf.model_path}/train/ours_{conf.iteration}/render/*_render.png")
test_render_paths = glob.glob(f"{conf.model_path}/test/ours_{conf.iteration}/render/*_render.png")

print(f"Number of images: {len(train_render_paths)} :: {len(test_render_paths)}")
assert len(train_render_paths) > 0
assert len(test_render_paths) > 0

train_data = []
for render_path in tqdm.tqdm(train_render_paths[:conf.max_images]):
    train_data.append(open_gbuffers(render_path))

test_data = []
for path in tqdm.tqdm(test_render_paths[:conf.max_images]):
    test_data.append(open_gbuffers(path))

# -------------------- Run training

for epoch in range(conf.num_epochs):
    random.shuffle(train_data)

    for gbuffer in train_data:
        output = model(gbuffer)

        optimizer.zero_grad()
        loss = criterion(output, gbuffer["gt"])
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{conf.num_epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        # Validate on the first test frame
        with torch.no_grad():
            output = model(train_data[0])
            print("train", psnr(train_data[0]["render"][0], train_data[0]["gt"][0]).mean(), psnr(output[0], train_data[0]["gt"][0]).mean())
            save_image(
                torch.cat([train_data[0]["render"], output, train_data[0]["gt"]], dim=0),
                f"output_train.ignore.jpg",
                nrow=1,
            )
        
        with torch.no_grad():
            output = model(test_data[0])
            print("test", psnr(test_data[0]["render"][0], test_data[0]["gt"][0]).mean(), psnr(output[0], test_data[0]["gt"][0]).mean())
            save_image(
                torch.cat([test_data[0]["render"], output, test_data[0]["gt"]], dim=0),
                f"output_test.ignore.jpg",
                nrow=1,
            )

        # if epoch % 5 == 0:
        #     print("Rendering video...")
        #     frames = []
        #     start_time = time.time()

        #     # Render the video
        #     for gbuffer in test_data:
        #         output = model(
        #             gbuffer["render"][None]).cuda()        #   .clamp(0, 1)
        #         frames.append(torch.cat([gbuffer["render"], output[0], gbuffer["gt"]], dim=2))

        #     end_time = time.time()
        #     elapsed_time = end_time - start_time
        #     fps = len(frames) / elapsed_time
        #     print(f"Rendered video at {fps:.2f} FPS")

        #     # Save to mp4
        #     torchvision.io.write_video(
        #         f"output.ignore.mp4",
        #         (torch.cat(frames) * 255).to(torch.uint8).moveaxis(1, -1),
        #         fps=30,
        #     )
        #     print("Video saved")


print("Training complete.")
