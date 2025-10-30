#!/bin/bash 
set -xe

# ============================================================
# Start
# ============================================================

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

python tools/compress_dataset.py

# Copy the rest of the files
SCENE_DIR="data/renders_compressed"
SCENE_LIST="shiny_kitchen shiny_bedroom shiny_livingroom shiny_office multichromeball_kitchen_v2 multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2"
for SCENE in $SCENE_LIST; do
    cp -r data/renders/$SCENE/sparse output/renders_compressed/$SCENE/sparse
    cp data/renders/$SCENE/transforms_train.json output/renders_compressed/$SCENE/transforms_train.json
    cp data/renders/$SCENE/transforms_test.json output/renders_compressed/$SCENE/transforms_test.json
done
