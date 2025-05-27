#!/bin/bash 
set -xe

# ============================================================
# OAR Script
# Run with: oarsub -S script.sh
# ============================================================

#OAR -q besteffort 
#OAR -l host=1/gpu=1,walltime=12:00:00
#OAR -p a40
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

module load cuda

# make use of a python torch environment
source ~/.bashrc
conda activate gausstracer
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";


# ============================================================
# Start
# ============================================================

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

RAYTRACER_VERSION="../optix-gaussian-raytracing/build/v74/"

RESOLUTION=512
SCENE_DIR="data/renders"
SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# SCENE_LIST="multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
OUTPUT_DIR="output/benchmark_v74"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --init_scale_factor 0.1 \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    python render_novel_views.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval \
        --raytracer_version $RAYTRACER_VERSION

    # Saving videos
    NOVEL_VIEWS_DIR=$OUTPUT_DIR/$SCENE/novel_views/ours_8000
    for BUFFER_DIR in $(find $NOVEL_VIEWS_DIR -mindepth 1 -maxdepth 1 -type d); do
        echo "Making video from: $BUFFER_DIR"
        ffmpeg -y -framerate 30 -pattern_type glob -i "$BUFFER_DIR/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$BUFFER_DIR.mp4"
    done

    ffmpeg -y \
        -i "$NOVEL_VIEWS_DIR/render.mp4" \
        -i "$NOVEL_VIEWS_DIR/diffuse.mp4" \
        -i "$NOVEL_VIEWS_DIR/glossy.mp4" \
        -i "$NOVEL_VIEWS_DIR/normal.mp4" \
        -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
        -map "[v]" \
        "$NOVEL_VIEWS_DIR/ours_$SCENE.mp4"

done
