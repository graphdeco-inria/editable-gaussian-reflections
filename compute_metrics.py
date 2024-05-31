import glob 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tyro
import os
from PIL import Image 
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_tensor
from dataclasses import dataclass
from tqdm import tqdm 
import pandas as pd 

#

@dataclass
class Conf:
    ground_truth_pattern: str = "output_full_views/{object}_{material}_{method}/00/test/ours_30000/gt/{i:05d}.png"
    predictions_pattern: str = "output_full_views/{object}_{material}_{method}/00/test/ours_30000/renders/{i:05d}.png"
    step: int = 1

conf = tyro.cli(Conf)

#
methods = ["ours", "baseline", "baseline_no_sh"]
metrics = dict(
    psnr=(PeakSignalNoiseRatio(data_range=1.0).cuda(), 2, True), 
    lpips=(LearnedPerceptualImagePatchSimilarity(normalize=True).cuda(), 3, False)
)
objects = ["sphere", "cube", "bunny"]
materials = ["smooth", "shiny", "metal", "mirror"]

#

#todo: highlight automatically


for label, (metric, rounding, higher_is_better) in metrics.items():
    scenes = [f"{object}_{material}" for material in materials for object in objects]
    df = pd.DataFrame(columns=scenes, index=methods).fillna(0)
    pbar = tqdm(total=len(scenes) * len(methods) * len(range(0, 100, conf.step)))
    for object in objects:
        for material in materials:
            for method in methods:
                for i in range(0, 100, conf.step):
                    scene = f"{object}_{material}"
                    gt = to_tensor(Image.open(conf.ground_truth_pattern.format(object=object, material=material, i=i, method=method)))
                    pred = to_tensor(Image.open(conf.predictions_pattern.format(object=object, material=material, i=i, method=method)))
                    df.loc[method, scene] += metric(pred[None].cuda(), gt[None].cuda()).item() / len(range(0, 100, conf.step))
                    pbar.update(1)
    
    def highlight(s):
        if higher_is_better:
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return [f"=={v:.{rounding}f}==" if max_val else f"{v:.{rounding}f}" for v, max_val in zip(s, is_best)]

    print(df.apply(highlight).to_markdown())
    print()