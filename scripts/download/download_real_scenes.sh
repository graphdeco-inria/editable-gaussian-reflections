#!/bin/bash
set -xe

DATA_DIR="./data"
mkdir -p $DATA_DIR

# Download datasets
# python tools/download_dataset.py --dataset refnerf
python tools/download_dataset.py --dataset neural_catacaustics

# Download priors for datasets
# DATASET_NAME="refnerf"
DATASET_NAME="neural_catacaustics"

pushd $DATA_DIR
    wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/priors/${DATASET_NAME}_priors.zip
    unzip ${DATASET_NAME}_priors.zip
    rm ${DATASET_NAME}_priors.zip
popd
