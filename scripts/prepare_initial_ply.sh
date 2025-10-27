#!/bin/bash 
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

DATA_DIR="./data"

DATASET_NAME="neural_catacaustics"
SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
RESOLUTION=128
SCALE=50

# DATASET_NAME=renders_priors
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# RESOLUTION=128
# SCALE=400

# DATASET_NAME=renders
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# RESOLUTION=128
# SCALE=400

for SCENE in $SCENE_LIST;
do
    python prepare_initial_ply.py \
        --source_path $DATA_DIR/$DATASET_NAME/$SCENE \
        --scale $SCALE \
        --resolution $RESOLUTION
done
