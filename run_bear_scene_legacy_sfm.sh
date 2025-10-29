
bash run_real_scene.sh output/neural_catacaustics/multibounce -s data/neural_catacaustics/multibounce --loss_weight_depth 0.0  --loss_weight_specular 0.01 --disable_znear_densif_pruning --init_type sfm "$@"

