#!/bin/bash 
set -xe

DATASET_NAME=neural_catacaustics
SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"

# DATASET_NAME=renders
# SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
# SCENE_LIST="multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"

for SCENE in $SCENE_LIST;
do
    VIDEO_PATH0=output/benchmark_${DATASET_NAME}_priors/$SCENE/novel_views/ours_8000/RENDER,DIFFUSE,GLOSSY,NORMAL.mp4
    VIDEO_PATH1=../EnvGS/data/novel_view/envgs/${DATASET_NAME}/envgs_$SCENE/RENDER,DIFFUSE,REFLECTION,NORMAL.mp4

    SAVE_DIR=output/benchmark_${DATASET_NAME}_priors/$SCENE/novel_views
    ffmpeg -y \
        -i $VIDEO_PATH0 \
        -i $VIDEO_PATH1 \
        -filter_complex "[0:v]scale=4716:768[v0];[1:v]scale=4716:768[v1];[v0][v1]vstack=inputs=2[v]" \
        -map "[v]" \
        "$SAVE_DIR/${SCENE}_compare.mp4"
done
