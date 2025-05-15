#!/bin/bash 
set -xe

# SCENE_DIR="data/real_datasets_v2_filmic/refnerf_priors"
# SCENE_LIST="gardenspheres sedan toycar"
# OUTPUT_DIR="output/benchmark_refnerf_priors"

# SCENE_DIR="data/real_datasets_v2_filmic/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# OUTPUT_DIR="output/benchmark_360_v2_priors"

# DATASET_NAME=neural_catacaustics
# SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"

DATASET_NAME=renders
# SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
SCENE_LIST="multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"

for SCENE in $SCENE_LIST;
do
    VIDEO_PATH0=output/output_512_filmic/benchmark_${DATASET_NAME}_priors/$SCENE/novel_views/RENDER,DIFFUSE,GLOSSY,NORMAL.mp4
    VIDEO_PATH1=output/benchmark_v74/$SCENE/novel_views/RENDER,DIFFUSE,GLOSSY,NORMAL.mp4
    VIDEO_PATH2=../EnvGS/data/novel_view/envgs/${DATASET_NAME}_prnormals/envgs_$SCENE/RENDER,DIFFUSE,REFLECTION,NORMAL.mp4
    VIDEO_PATH3=../EnvGS/data/novel_view/envgs/${DATASET_NAME}_gtnormals/envgs_$SCENE/RENDER,DIFFUSE,REFLECTION,NORMAL.mp4

    COMPARE_DIR=output/output_512_filmic/benchmark_${DATASET_NAME}_priors/$SCENE/novel_views
    ffmpeg -y -i $VIDEO_PATH0 -i $VIDEO_PATH1 -i $VIDEO_PATH2 -i $VIDEO_PATH3 -filter_complex "[0:v]scale=4716:768[v0];[1:v]scale=4716:768[v1];[2:v]scale=4716:768[v2];[3:v]scale=4716:768[v3];[v0][v1][v2][v3]vstack=inputs=4[v]" -map "[v]" "$COMPARE_DIR/${SCENE}_compare.mp4"
done
