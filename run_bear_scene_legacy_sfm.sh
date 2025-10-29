
bash run_real_scene.sh output/neural_catacaustics/multibounce -s data/neural_catacaustics/multibounce --loss_weight_depth 0.1     --loss_weight_specular 0.01 --init_scale 1.5 --disable_znear_densif_pruning --init_type sfm "$@"

