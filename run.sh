set -e

python train.py -m "$@"
python render.py -m "$@"
if [ -z "$SKIP_EVAL" ]; then
    python metrics.py -m "$1"
    python measure_fps.py -m "$1"
fi