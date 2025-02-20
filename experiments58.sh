

for scene in shiny_bedroom shiny_kitchen shiny_office shiny_livingroom; do
    bash run.sh --raytracer_version v15_1bounce_noblur_ignore0vol_jitter -s colmap/${scene} --glossy_loss_weight 0.001 -m output_exp57/best_state_${scene}_no_znear --glossy_bbox_size_mult 8.0 --no_znear_init_pruning --no_znear_densif_pruning 
done


for scene in shiny_bedroom shiny_kitchen shiny_office shiny_livingroom; do
    bash run.sh --raytracer_version v15_1bounce_noblur_ignore0vol_jitter -s colmap/${scene} --glossy_loss_weight 0.001 -m output_exp57/best_state_${scene}_bbox12 --glossy_bbox_size_mult 12.0  
done
