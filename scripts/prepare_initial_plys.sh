#!/bin/bash 
set -xe

# ============================================================
# Start
# ============================================================

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

# SCENE_DIR="data/renders"
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# RESOLUTION=128
# VOXEL_SCALE=400

# SCENE_DIR="data/renders_compressed"
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# RESOLUTION=128
# VOXEL_SCALE=400

# SCENE_DIR="data/renders_predicted"
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# RESOLUTION=128
# VOXEL_SCALE=200

# SCENE_DIR="data/refnerf"
# SCENE_LIST="gardenspheres sedan toycar"
# RESOLUTION=256
# VOXEL_SCALE=50

SCENE_DIR="data/neural_catacaustics"
SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
RESOLUTION=128
VOXEL_SCALE=50

# SCENE_DIR="data/360_v2"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# RESOLUTION=256
# VOXEL_SCALE=50

for SCENE in $SCENE_LIST;
do
    python prepare_initial_ply.py \
        --source_path $SCENE_DIR/$SCENE \
        --resolution $RESOLUTION \
        --do_depth_fit \
        --voxel_scale $VOXEL_SCALE
done
