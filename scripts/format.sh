#!/bin/bash
set -xe

find gaussian_tracing/cuda \
  -path "*/third_party/*" -prune -o \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i {} \;

# Currently only for select files and directories
SELECT_PATHS="gaussian_tracing/ tests/ setup.py train.py render.py render_novel_views.py prepare_initial_ply.py measure_fps.py"
ruff check --extend-select I --fix $SELECT_PATHS
ruff format --line-length 200 $SELECT_PATHS
