#!/bin/bash
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

DATASET_DIR="./data/renders"
OUTPUT_DIR="./output/benchmark_v52"
RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v52/"

SCENE_LIST="shiny_bedroom shiny_kitchen shiny_livingroom shiny_office"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $DATASET_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r 768 \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $DATASET_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r 768 \
        --raytracer_version $RAYTRACER_VERSION
done
