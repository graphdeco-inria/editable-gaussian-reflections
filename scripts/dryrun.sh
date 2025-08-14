#!/bin/bash
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

RESOLUTION=256
SCENE_DIR="data/renders"
SCENE_LIST="shiny_kitchen"
OUTPUT_DIR="output/dryrun"

for SCENE in $SCENE_LIST;
do
    python train.py \
        --source_path $SCENE_DIR/$SCENE \
        --model_path $OUTPUT_DIR/$SCENE \
        --resolution $RESOLUTION \
        --eval \
        --max_images 2 \
        --save_iterations 50 \
        --test_iterations 50 \
        --iterations 100

    python render.py \
        --source_path $SCENE_DIR/$SCENE \
        --model_path $OUTPUT_DIR/$SCENE \
        --resolution $RESOLUTION \
        --max_images 2
done
