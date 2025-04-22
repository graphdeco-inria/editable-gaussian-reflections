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
class NetworkConfig:
    lr: float = 1e-3
    num_epochs: int = 100 
    model_path: str = "output_v74/shiny_kitchen"
    load_iteration: int = 7500
    max_images: Optional[int] = None
    kernel_size: int = 1
    pe: int = 6
    width: int = 512
    num_layers: int = 5
    resnet: bool = False
    epoch_size: int = 100
    downsampling: int = 1

conf = tyro.cli(Config)

# todo try a small sweep on positional encoding 

# todo add label to save variants of mlp
# todo save pt in the model_path
# todo make video render work again

# todo concat all rgb maps
# todo concat incident radiance, brdf maps
# todo include reflection ray direction under a flag
# todo find a faster network that still works (maybe just some conv layers?)
# todo try timm
# todo sanity check train on the maps ONLY as input, is it just memorizing the data?

# todo try a timm network 
# todo try layer sizes again 

# for a single view with width 512 and NO POSITION: Iteration [7500], Loss: 0.0002, Time: 00:12:02, Train PSNR: 36.07, Test PSNR: 21.22

# ------------------- Define model

def positional_encoding(positions, freqs=conf.pe): # taken from tensorf
    freq_bands = (2**torch.arange(freqs).to(torch.bfloat16)).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, conf.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"), 
            nn.GroupNorm(1, conf.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same")
        )

    def forward(self, x):
        return (x + self.layers(x)) / math.sqrt(2)
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        if conf.resnet:
            self.layers = nn.Sequential(
                nn.LazyConv2d(conf.width, kernel_size=1),
                ResBlock(),
                ResBlock(),
                nn.Conv2d(conf.width, 3, kernel_size=1)
            ).cuda()
        else:
            layers = [
                nn.LazyConv2d(conf.width, kernel_size=conf.kernel_size, padding="same"),
                nn.ReLU()
            ]
            for _ in range(conf.num_layers - 2):
                layers += [
                    nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"),
                    nn.ReLU()
                ]
            layers += [nn.Conv2d(conf.width, 3, kernel_size=conf.kernel_size, padding="same")]
            self.layers = nn.Sequential(*layers).cuda()

    def forward(self, gbuffer):
        inputs = torch.cat([
            gbuffer["render"],
            gbuffer["diffuse"],
            gbuffer["glossy"],
            gbuffer["normal"],
            gbuffer["F0"],
            gbuffer["position"],
            gbuffer["roughness"],
        ], dim=1)
        inputs_pe = positional_encoding(inputs.moveaxis(1, -1)).moveaxis(-1, 1)
        return self.layers(inputs_pe)

model = Model().to(torch.bfloat16)

model.train()
optimizer = optim.Adam(model.parameters(), lr=conf.lr) 
criterion = torch.nn.MSELoss()
transform = transforms.Compose(
    [
        transforms.Resize(
            (512 // conf.downsampling, 768 // conf.downsampling),
            antialias=True,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
)

# -------------------- Load training data

print(f"{conf.model_path}/train/ours_{conf.load_iteration}")

def open_image(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).cuda()[None].to(torch.bfloat16)

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

train_render_paths = glob.glob(f"{conf.model_path}/train/ours_{conf.load_iteration}/render/*_render.png")
test_render_paths = glob.glob(f"{conf.model_path}/test/ours_{conf.load_iteration}/render/*_render.png")

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

start = time.time()
iteration = 0 

while True:
    random.shuffle(train_data)
    
    epoch_done = False
    while not epoch_done:
        for gbuffer in train_data:
            output = model(gbuffer)
            optimizer.zero_grad()
            loss = criterion(output, gbuffer["gt"])
            loss.backward()
            optimizer.step()
            iteration += 1

            elapsed = time.time() - start

            if iteration % conf.epoch_size == 0:
                epoch_done = True
                break

    with torch.no_grad():
        # Validate on the first test frame
        with torch.no_grad():
            output = model(train_data[0])
            train_psnr = psnr(output[0], train_data[0]["gt"][0]).mean()
            save_image(
                torch.cat([train_data[0]["render"], output, train_data[0]["gt"]], dim=0),
                f"{conf.model_path}/cnn_prediction_train.png",
                nrow=1,
            )
        
        with torch.no_grad():
            output = model(test_data[0])
            test_psnr = psnr(output[0], test_data[0]["gt"][0]).mean()
            save_image(
                torch.cat([test_data[0]["render"], output, test_data[0]["gt"]], dim=0),
                f"{conf.model_path}/cnn_prediction_test.png",
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
    
    dt = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"Iteration [{iteration}], Loss: {loss.item():.04f}, Time: {dt}, Train PSNR: {train_psnr.item():.02f}, Test PSNR: {test_psnr:.02f}")

print("Training complete.")
