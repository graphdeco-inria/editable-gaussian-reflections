#!/bin/bash
set -xe

# Used to load fallback exr files
export OPENCV_IO_ENABLE_OPENEXR=1

RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v52/"

SCENE_DIR="data/shiny_dataset_priors"
SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
OUTPUT_DIR="output/benchmark_shiny_dataset_priors"

# SCENE_DIR="data/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# OUTPUT_DIR="output/benchmark_360_v2_priors"

# SCENE_DIR="data/refnerf_priors"
# SCENE_LIST="gardenspheres sedan toycar"
# OUTPUT_DIR="output/benchmark_refnerf_priors"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r 768 \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r 768 \
        --raytracer_version $RAYTRACER_VERSION
done
