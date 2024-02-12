: ${1?specify a dataset path}

set -e

timestamp=$(date -Is)
label=$(basename $1)$LABEL/$timestamp
python train.py --label $label -s "${@}"
python render.py --skip_train -m output/$label --eval
ffmpeg  -i output/$label/test/ours_30000/renders/%05d.png -i output/$label/test/ours_30000/gt/%05d.png -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p output/$label/test.mp4
