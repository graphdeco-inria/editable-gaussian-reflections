#!/bin/bash
set -xe

rm -rf build
rm -rf gaussian_tracing/cuda/build

pip install -v -e .[dev]
