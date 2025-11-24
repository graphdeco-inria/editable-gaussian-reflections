#!/bin/bash
set -xe

DATA_DIR="./data"
mkdir -p $DATA_DIR

# Download datasets
python tools/download_dataset.py --dataset renders_compressed
# python tools/download_dataset.py --dataset renders_predicted
