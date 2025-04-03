#!/bin/bash
set -xe

# Currently only for scene/ directory
# ruff check --extend-select I --fix scene/
ruff format scene/
