#!/bin/bash
set -xe

# Currently only for select files and directories
SELECT_PATHS="gaussian_tracing/ examples/ tests/ setup.py"
ruff check --extend-select I --fix $SELECT_PATHS
ruff format $SELECT_PATHS
