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
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")

@dataclass
class NetworkConfig:
    lr: float = 2e-4 #! higher seemed problematic for cnn
    num_epochs: int = 100 
    model_path: str = "output_v74/shiny_kitchen"
    load_iteration: int = 7500
    max_images: Optional[int] = None
    kernel_size: int = 3
    pe_rgb: int = 8
    pe_brdf: int = 8
    pe_geo: int = 8
    width: int = 128 # more seemed better for mlp, but higher fails CNN. review init
    num_layers: int = 5 # more seemed better, review
    use_norm: bool = False  # slows things down, especially groupnorm but instancenorm also
    epoch_size: int = 100
    downsampling: int = 1
    batch_size: int = 1
    render_video_interval: int = 100

    use_tcnn: bool = False 
    use_diffusers: bool = False
    lpips_loss: bool = False

conf = tyro.cli(NetworkConfig)

# todo include reflection ray direction under a flag. IMO largest possible gain 
# todo try the diffusers network again 

# todo does better init (kaiming) fix 512 width training?
# todo try resnet again to be sure, does it fix 512 training?

# todo switch to rendering on the fly
# todo concat incident radiance, brdf maps
# todo concat all bounces separately (not just glossy)

# ------------------- Define model

def positional_encoding(positions, freqs): # taken from tensorf
    freq_bands = (2**torch.arange(freqs).to(torch.bfloat16)).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        if conf.use_tcnn:
            assert conf.batch_size == 1
            import tinycudann as tcnn
            self.layers = tcnn.NetworkWithInputEncoding(n_input_dims=21, n_output_dims=3, encoding_config={}, network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": conf.num_layers - 2
            }).cuda()
        elif conf.use_diffusers:
            import diffusers
            self.layers = diffusers.models.UNet2DModel(
                sample_size=32,       
                in_channels=336,          
                out_channels=3,         
                layers_per_block=2,
                block_out_channels=(32, 64),  
                down_block_types=("DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D"),
            ).cuda()
        else:
            layers = [
                nn.LazyConv2d(conf.width, kernel_size=conf.kernel_size, padding="same"),
                nn.ReLU()
            ]
            for _ in range(conf.num_layers - 2):
                if conf.use_norm:
                    layers += [nn.InstanceNorm2d(conf.width)]
                layers += [
                    nn.Conv2d(conf.width, conf.width, kernel_size=conf.kernel_size, padding="same"),
                    nn.ReLU()
                ]
            layers += [nn.Conv2d(conf.width, 3, kernel_size=conf.kernel_size, padding="same")]
            self.layers = nn.Sequential(*layers).cuda()

    def forward(self, gbuffer):
        rgb_inputs = torch.cat([
            gbuffer["render"], 
            gbuffer["diffuse"], 
            gbuffer["glossy"], 
        ], dim=1)
        brdf_inputs = torch.cat([
            gbuffer["F0"],
            gbuffer["roughness"],
        ], dim=1)
        geometry_inputs = torch.cat([
            gbuffer["normal"],
            gbuffer["position"]
        ], dim=1)

        if conf.use_tcnn:
            input_maps = torch.cat([ rgb_inputs, brdf_inputs, geometry_inputs ], dim=1)
            return self.layers(input_maps.flatten(2, 3).moveaxis(-1, 1)[0].half()).moveaxis(0, -1).reshape(1, 3, input_maps.shape[-2], input_maps.shape[-1]).to(torch.bfloat16)
        elif conf.use_diffusers:
            rgb_inputs_pe = positional_encoding(rgb_inputs.moveaxis(1, -1), conf.pe_rgb).moveaxis(-1, 1)
            brdf_inputs_pe = positional_encoding(brdf_inputs.moveaxis(1, -1), conf.pe_brdf).moveaxis(-1, 1)
            geometry_inputs_pe = positional_encoding(geometry_inputs.moveaxis(1, -1), conf.pe_geo).moveaxis(-1, 1)
            return self.layers(torch.cat([ rgb_inputs_pe, brdf_inputs_pe, geometry_inputs_pe ], dim=1), torch.zeros(rgb_inputs_pe.shape[0], device=rgb_inputs_pe.device)).sample
        else:
            rgb_inputs_pe = positional_encoding(rgb_inputs.moveaxis(1, -1), conf.pe_rgb).moveaxis(-1, 1)
            brdf_inputs_pe = positional_encoding(brdf_inputs.moveaxis(1, -1), conf.pe_brdf).moveaxis(-1, 1)
            geometry_inputs_pe = positional_encoding(geometry_inputs.moveaxis(1, -1), conf.pe_geo).moveaxis(-1, 1)
            return self.layers(torch.cat([ rgb_inputs_pe, brdf_inputs_pe, geometry_inputs_pe ], dim=1))

# -------------------- Load training data

print(f"{conf.model_path}/train/ours_{conf.load_iteration}")

def open_image(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).cuda().to(torch.bfloat16)

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
for path in tqdm.tqdm(sorted(test_render_paths[:conf.max_images])):
    test_data.append(open_gbuffers(path))

# -------------------- Create network

model = Model().to(torch.bfloat16)

model.train()
optimizer = optim.Adam(model.parameters(), lr=conf.lr) 

if conf.lpips_loss:
    import lpips

    class Criterion:
        def __init__(self):
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            self.l1 = torch.nn.L1Loss()
        
        def __call__(self, x, y):
            return self.l1(x, y) + self.lpips(x / 2 + 0.5, y / 2 + 0.5)
    
    criterion = Criterion()
else:
    criterion = torch.nn.MSELoss()

# -------------------- Run training

start = time.time()
iteration = 0 
epoch = 0

while True:
    random.shuffle(train_data)
    
    epoch_done = False
    while not epoch_done:
        for gbuffer in DataLoader(train_data, batch_size=conf.batch_size):
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
    epoch += 1

    with torch.no_grad():
        with torch.no_grad():
            output = model(next(iter(DataLoader(train_data, batch_size=1, shuffle=False))))
            train_psnr = psnr(output[0], train_data[0]["gt"]).mean()
            save_image(
                torch.stack([train_data[0]["render"], output[0], train_data[0]["gt"]], dim=0),
                f"{conf.model_path}/cnn_prediction_train.png",
                nrow=1,
                padding=0
            )
        
        with torch.no_grad():
            output = model(next(iter(DataLoader(test_data, batch_size=1, shuffle=False))))
            test_psnr = psnr(output[0], test_data[0]["gt"]).mean()
            save_image(
                torch.stack([test_data[0]["render"], output[0], test_data[0]["gt"]], dim=0),
                f"{conf.model_path}/cnn_prediction_test.png",
                nrow=1,
                padding=0
            )

        if epoch % conf.render_video_interval == 0 and (conf.max_images is None or conf.max_images > 5):
            print("Rendering video...")
            frames = []
            start_time = time.time()

            for gbuffer in DataLoader(test_data, batch_size=1, shuffle=False):
                output = model(gbuffer).cuda().clamp(0, 1)
                frames.append(torch.cat([gbuffer["render"][0], output[0], gbuffer["gt"][0]], dim=2).detach().cpu())

            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = len(frames) / elapsed_time
            print(f"Rendered video at {fps:.2f} FPS")

            for (crf, label) in [("18", "hq"), ("30", "lq")]:
                torchvision.io.write_video(
                    f"{conf.model_path}/cnn_test_video_{label}.mp4",
                    (torch.stack(frames) * 255).to(torch.uint8).moveaxis(1, -1),
                    fps=30,
                    options={"crf": crf, "preset": "slow"}
                )
            print("Video saved")
    
    dt = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"Iteration [{iteration}], Loss: {loss.item():.04f}, Time: {dt}, Train PSNR: {train_psnr.item():.02f}, Test PSNR: {test_psnr:.02f}")

    torch.save(model, f"{conf.model_path}/model.pt")

print("Training complete.")
