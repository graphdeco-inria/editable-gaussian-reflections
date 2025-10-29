
for scene in data/neural_catacaustics/*; do 
    bash run_real_scene.sh ${scene/data/output} -s $scene --loss_weight_depth 0.0 --init_scale 0.1 --loss_weight_specular 0.01 --disable_znear_densif_pruning "$@"
done
