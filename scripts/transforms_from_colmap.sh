

SCENE=${1?Please provide scene path as first argument}

colmap model_converter \
    --input_path "$SCENE/sparse/0" \
    --output_path "$SCENE/sparse/0" \
    --output_type TXT

python tools/colmap2nerf.py \
    --images "$SCENE/images" \
    --text "$SCENE/sparse/0" \
    --out "$SCENE/transforms_train.json" \
    --keep_colmap_coords

rm $SCENE/sparse/0/*.txt