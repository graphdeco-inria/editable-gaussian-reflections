#!/bin/bash
set -xe

find gaussian_tracing/cuda \
  -path "*/third_party/*" -prune -o \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i {} \;

# Currently only for select files and directories
SELECT_PATHS="gaussian_tracing/ examples/ tests/ setup.py"
ruff check --extend-select I --fix $SELECT_PATHS
ruff format $SELECT_PATHS
