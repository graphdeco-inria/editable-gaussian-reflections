#!/bin/bash
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

RESOLUTION=512
RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v52/"

SCENE_DIR="data/renders"
SCENE_LIST="shiny_kitchen shiny_livingroom shiny_office shiny_bedroom"
OUTPUT_DIR="output/benchmark_v52"

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
        --skip_video \
        --raytracer_version $RAYTRACER_VERSION
done
