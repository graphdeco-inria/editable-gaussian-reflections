#!/bin/bash 
set -xe

# ============================================================
# Start
# ============================================================

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

DATA_DIR="./data/real_datasets_v3_filmic"
OUTPUT_DIR="./output/plys"

# SCENE_DIR="data/real_datasets_v3_filmic/neural_catacaustics_priors"
# SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
# RESOLUTION=128
# SCALE=50

# SCENE_DIR="data/real_datasets_v3_filmic/refnerf_priors"
# SCENE_LIST="gardenspheres sedan toycar"
# RESOLUTION=256
# SCALE=50

# SCENE_DIR="data/real_datasets_v3_filmic/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# RESOLUTION=256
# SCALE=50

# SCENE_DIR="renders_priors"
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# RESOLUTION=128
# SCALE=200

DATA_DIR="./data"
DATASET_NAME=renders
SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
RESOLUTION=128
SCALE=400

for SCENE in $SCENE_LIST;
do
    python prepare_initial_ply.py \
        --source_path $DATA_DIR/$DATASET_NAME/$SCENE \
        --output_dir $OUTPUT_DIR/$DATASET_NAME/$SCENE \
        --scale $SCALE \
        --resolution $RESOLUTION
done
