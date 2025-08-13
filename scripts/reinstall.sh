#!/bin/bash
set -xe

rm -rf build
rm -rf gaussian_tracing/cuda/build
rm gaussian_tracing/libgausstracer.so

pip install -v -e .[dev]
