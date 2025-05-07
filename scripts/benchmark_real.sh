#!/bin/bash 
set -xe

# ============================================================
# OAR Script
# Run with: oarsub -S script.sh
# ============================================================

#OAR -q besteffort 
#OAR -l host=1/gpu=1,walltime=12:00:00
#OAR -p a40
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

module load cuda

# make use of a python torch environment
source ~/.bashrc
conda activate gausstracer
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";


# ============================================================
# Start
# ============================================================

RESOLUTION=512
RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v74/"

SCENE_DIR="data/real_datasets_v2_filmic/refnerf_priors"
SCENE_LIST="gardenspheres sedan toycar"
OUTPUT_DIR="output/benchmark_refnerf_priors"

# SCENE_DIR="data/real_datasets_v2_filmic/neural_catacaustics_priors"
# SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
# OUTPUT_DIR="output/benchmark_neural_catacaustics_priors"

# SCENE_DIR="data/real_datasets_v2_filmic/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# OUTPUT_DIR="output/benchmark_360_v2_priors"

# SCENE_DIR="data/real_datasets_v2_filmic/renders_priors"
# SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
# # SCENE_LIST="multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# OUTPUT_DIR="output/benchmark_renders_priors"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --no_znear_densif_pruning \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    python render_novel_views.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    # Saving videos
    IMAGES_DIR=$OUTPUT_DIR/$SCENE/novel_views
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/ours_6000/render/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$IMAGES_DIR/RENDER.mp4"
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/ours_6000/diffuse/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$IMAGES_DIR/DIFFUSE.mp4"
    ffmpeg -y -framerate 30 -pattern_type glob -i "$IMAGES_DIR/ours_6000/glossy/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$IMAGES_DIR/GLOSSY.mp4"
    ffmpeg -y -i "$IMAGES_DIR/RENDER.mp4" -i "$IMAGES_DIR/DIFFUSE.mp4" -i "$IMAGES_DIR/GLOSSY.mp4" -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" "$IMAGES_DIR/RENDER,DIFFUSE,GLOSSY.mp4"

done
