set -e

python train.py -m "$@"
bash render_novel_views.sh "$@"

