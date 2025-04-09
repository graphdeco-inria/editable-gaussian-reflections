#!/bin/bash
set -xe

export LOAD_FROM_IMAGE_FILES=1
export OPENCV_IO_ENABLE_OPENEXR=1

pytest -s tests/
