#!/bin/bash
set -xe

find editable_gauss_refl/cuda \
  -path "*/third_party/*" -prune -o \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i {} \;

# Currently only for select files and directories
SELECT_PATHS="editable_gauss_refl/ tests/ setup.py train.py render.py tools/render_novel_views.py prepare_initial_ply.py measure_fps.py"
ruff check --extend-select I --fix $SELECT_PATHS
ruff format --line-length 200 $SELECT_PATHS
