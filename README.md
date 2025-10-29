# Editable Physically-based Reflections in Raytraced Gaussian Radiance Fields 

Yohan Poirier-Ginter, Jeffrey Hu, Jean-François Lalonde, George Drettakis

[Webpage](https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/) | [Paper](https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/content/paper.pdf) | [Video](https://www.youtube.com/watch?v=Ag9xM1Zm0AY) | [Other GRAPHDECO Publications](http://www-sop.inria.fr/reves/publis/gdindex.php) | [NERPHYS project page](https://project.inria.fr/nerphys/) <br>
<!-- todo dataset download link, scene download link -->

![Teaser image](assets/teaser.png)

## Installation

We have included the official OptiX SDK header files in the third_party/optix directory as a submodule, so by default, you don't need to download the OptiX SDK from the NVIDIA official website, just add the `--recursive` flag when cloning the repository.

The default installation is as follows:

```bash
git clone git@gitlab.inria.fr:ypoirier/gaussian-splatting-raytraced.git --recursive
cd gaussian-splatting-raytraced

conda create -n editable_gauss_refl python=3.12
conda activate editable_gauss_refl

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

To build the cuda raytracer run
```bash
bash make.sh
```
alternatively you can pip install this project.

To test if installed correctly, run
```bash
bash ./scripts/test.sh
bash ./scripts/dryrun.sh
```

### Troubleshooting
If you run into cmake or gcc version issues, try using conda to install newer versions.

```bash
conda install -c conda-forge cxx-compiler==1.6.0 -y
conda install anaconda::cmake -y
```

## Viewing pretrained models
<!-- todo first download-->

```bash 
bash gaussian_viewer.py -m <model_path>
```
e.g. `bash gaussian_viewer.py -m output/neural_catacaustics/bear`.

## Downloading the datasets
<!-- todo -->

## Running the scenes 
To run the synthetic scenes with ground truth inputs (`data/renders`):
```bash 
bash run_all_synthetic.sh
```

To run extra synthetic scenes used in demos and examples (chromeball and book scenes, `data/demos`):
```bash 
bash run_all_demos.sh
```

To run the synthetic scenes with network inputs (`data/renders_priors`):
```bash 
bash run_all_synthetic.sh
```
(As pointed out in the paper, we obtain rather poor results in these scenes)

To run the real scenes from the Neural Catacaustics dataset: 
```bash 
bash run_all_neural_catacaustics.sh
```
Note that in the real scenes, depth regularization was disabled since it did not improve results.

The bear scene shown in the video was run on an older configuration which still use SfM init, provided for completeness: () 
```
bash run_bear_scene_legacy_sfm.sh
```
<!-- todo: editing -->

## Detailled commands

### Predicting network priors
Refer to the auxillary codebase
https://github.com/jefequien/GenPrior/tree/main/tools.


### Creating the initial point cloud 

```bash 
python prepare_initial_ply.py --mode <sfm|dense>
```
We recommend working with dense init; this code base does not support densification.

You may need to adjust the `--binning_scale` flag to get good results.

### Training 

```bash
SCENE_PATH=data/renders/shiny_kitchen
MODEL_PATH=out/shiny_kitchen
python train.py -s $SCENE_PATH -m $MODEL_PATH
```

### Rendering
```bash
SCENE_PATH=data/renders/shiny_kitchen
MODEL_PATH=out/shiny_kitchen
SPP=128 # samples per pixel
python render.py -s $SCENE_PATH -m $MODEL_PATH --spp $SPP
```
We rendered at 128spp for evaluation but lower values can give adequate results.

You may need to adjust the near clipping plane with the `--znear` flag if you run other scenes.

To render a view of the reconstructed environment, pass in the flag `--modes env_rot_1`.

### Evaluation

<!-- todo mention why psnr has improved a bit-->
PSNR values might be slightly higher than reported in the paper (on average) since we fixed an minor aliasing issue with the training data. 

### Measuring framerates
```bash
MODEL_PATH=out/shiny_kitchen
python measure_fps.py -m $MODEL_PATH
```

## Using the interactive viewer
<!-- todo
```bash 
MODEL_PATH=out/shiny_kitchen
python gaussian_viewer.py local $MODEL_PATH 8000
``` -->



