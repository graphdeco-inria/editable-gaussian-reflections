#!/bin/bash
set -xe

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
done
