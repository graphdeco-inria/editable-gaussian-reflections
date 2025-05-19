#!/bin/bash 
set -xe

SCENE_DIR="data/real_datasets_v2_filmic/neural_catacaustics_priors"
# SCENE_LIST="compost concave_bowl2 crazy_blade2 hallway_lamp multibounce silver_vase2 wateringcan2"
SCENE_LIST="compost"

for SCENE in $SCENE_LIST;
do
    python prepare_initial_ply.py \
        --source_path $SCENE_DIR/$SCENE
done
