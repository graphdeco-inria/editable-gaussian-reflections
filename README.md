# Editable Physically-based Reflections in Raytraced Gaussian Radiance Fields 

## Installation

We have included the official OptiX SDK header files in the third_party/optix directory as a submodule, so by default, you don't need to download the OptiX SDK from the NVIDIA official website, just add the --recursive flag when cloning the repository.

The default installation is as follows:

```bash
git clone git@gitlab.inria.fr:ypoirier/gaussian-splatting-raytraced.git --recursive
cd gaussian-splatting-raytraced

conda create -n gausstracer python=3.12
conda activate gausstracer

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install -v -e .[dev]
```

To test if installed correctly, run
```bash
bash ./scripts/test.sh
bash ./scripts/dryrun.sh
```

If you are modifying the raytracer you cna rebuild easily with
```bash
bash make.sh
```

## Debugging
If you run into cmake or gcc version issues, try using conda to install newer versions.

```bash
conda install -c conda-forge cxx-compiler==1.6.0 -y
conda install anaconda::cmake -y
```

## Training 

```bash
SCENE=data/renders/shiny_kitchen
OUTDIR=out/shiny_kitchen
python render.py -s $SCENE -m $OUTDIR
```

## Rendering
```bash
SCENE=data/renders/shiny_kitchen
OUTDIR=out/shiny_kitchen
SPP=128
python render.py -s $SCENE -m $OUTDIR --spp $SPP
```
We rendered at 128spp for evaluation but lower values can give adequate results.

To render a view of the reconstructed environment, pass in the flag `--modes env_rot_1`.

## Evaluation
<!-- todo -->

<!-- ## Measuring FPS
```bash
SCENE=data/renders/shiny_kitchen
OUTDIR=out/shiny_kitchen
SPP=128
python measure_fps.py -s $SCENE -m $OUTDIR
``` -->

## Using the interactive viewer
<!-- todo -->
```bash 
OUTDIR=out/shiny_kitchen
python gaussianviewer.py local $OUTDIR 8000
```

## Creating the dense init point cloud
<!-- todo -->

<!-- todo note that the specular pass is called glossy -->

<!-- ------------------------------------------ -->

