set -e
python train.py -m "${@}"
python render.py -m "${@}"
python metrics.py -m "$1"
python measure_fps.py -m "$1"