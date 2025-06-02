
if [[ -n "${RENDER_ONLY}" ]]; then
    ZNEAR=1.0 python render.py "${@}"
else
    python train.py "${@}" && 
    ZNEAR=1.0 python render.py "${@}" && 
    ZNEAR=1.0 python measure_fps.py "${@}"
fi &&
true