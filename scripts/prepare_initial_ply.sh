#!/bin/bash 
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

DATA_DIR="./data"

DATASET_NAME="neural_catacaustics"
SCENE_LIST="compost hallway_lamp multibounce"
RESOLUTION=128
VOXEL_SCALE=50

# DATASET_NAME=renders_priors
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# RESOLUTION=128
# VOXEL_SCALE=400

# DATASET_NAME=demos
# SCENE_LIST="data/demos/multichromeball data/demos/multichromeball_identical data/demos/multichromeball_tint data/demos/shiny_office_with_book"
# RESOLUTION=128
# VOXEL_SCALE=400

# DATASET_NAME=renders
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# RESOLUTION=128
# VOXEL_SCALE=400

for SCENE in $SCENE_LIST;
do
    python prepare_initial_ply.py \
        --source_path $DATA_DIR/$DATASET_NAME/$SCENE \
        --resolution $RESOLUTION \
        --voxel_scale $VOXEL_SCALE 
done
