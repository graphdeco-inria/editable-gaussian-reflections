#!/bin/bash
set -xe

rm -rf build
rm -rf editable_gauss_refl/cuda/build

pip install -v -e .[dev]
