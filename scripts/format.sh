#!/bin/bash
set -xe

# Currently only for select files and directories
SELECT_PATHS="gaussian_tracing/ train.py render.py render_novel_views.py scene/ tests/"
ruff check --select I --fix $SELECT_PATHS
ruff format $SELECT_PATHS
