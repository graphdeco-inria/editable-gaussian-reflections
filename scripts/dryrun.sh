#!/bin/bash
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

RESOLUTION=256
RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v52/"

SCENE_DIR="data/renders"
SCENE_LIST="shiny_kitchen"
OUTPUT_DIR="output/dryrun"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --max_images 2 \
        --save_iterations 50 \
        --test_iterations 50 \
        --iterations 100 \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --max_images 2 \
        --skip_video \
        --raytracer_version $RAYTRACER_VERSION
done
