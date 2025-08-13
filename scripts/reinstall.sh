#!/bin/bash
set -xe

rm -rf build
rm -rf gaussian_tracing/cuda/build
rm -f gaussian_tracing/libgausstracer.so

pip install -v -e .[dev]
