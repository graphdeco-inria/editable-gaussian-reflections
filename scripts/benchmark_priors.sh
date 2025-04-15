#!/bin/bash
set -xe

RESOLUTION=512
RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v52/"

SCENE_DIR="data/shiny_dataset_priors"
SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
OUTPUT_DIR="output/benchmark_shiny_dataset_priors"

# SCENE_DIR="data/refnerf_priors"
# SCENE_LIST="gardenspheres sedan toycar"
# OUTPUT_DIR="output/benchmark_refnerf_priors"

# SCENE_DIR="data/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# OUTPUT_DIR="output/benchmark_360_v2_priors"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --num_farfield_init_points 100_000 \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --raytracer_version $RAYTRACER_VERSION
done
