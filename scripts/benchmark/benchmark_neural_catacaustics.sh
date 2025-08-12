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

# make use of a python torch environment
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))";


# ============================================================
# Start
# ============================================================

# RESOLUTION=512
# SCENE_DIR="data/real_datasets_v3_filmic/refnerf_priors"
# SCENE_LIST="gardenspheres sedan toycar"
# OUTPUT_DIR="output/benchmark_refnerf_priors"

RESOLUTION=512
SCENE_DIR="data/real_datasets_v3_filmic/neural_catacaustics_priors"
SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
OUTPUT_DIR="output/benchmark_neural_catacaustics_priors"

# RESOLUTION=512
# SCENE_DIR="data/real_datasets_v3_filmic/360_v2_priors"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
# OUTPUT_DIR="output/benchmark_360_v2_priors"

# RESOLUTION=512
# SCENE_DIR="data/real_datasets_v3_filmic/renders_priors"
# SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office"
# # SCENE_LIST="multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
# OUTPUT_DIR="output/benchmark_renders_priors"

for SCENE in $SCENE_LIST;
do
    python train.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --init_scale_factor 0.1 \
        --no_znear_densif_pruning \
        --position_loss_weight 0.0 \
        --eval

    python render.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval

    ZNEAR=0.5 python render_novel_views.py \
        -s $SCENE_DIR/$SCENE \
        -m $OUTPUT_DIR/$SCENE \
        -r $RESOLUTION \
        --eval

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
