python train.py -m "${@}" &&
ZNEAR=1.0 python render.py -m "${@}" && 
python metrics.py -m "$1" &&
python measure_fps.py -m "$1" &&
true

