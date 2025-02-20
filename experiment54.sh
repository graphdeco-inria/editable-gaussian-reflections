

NO_TONEMAPPING=1 bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/baseline_with_densif_pruning --no_znear_init_pruning

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/tonemapped_with_densif_pruning --no_znear_init_pruning

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/tonemapped+init_pruning_with_densif_pruning

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/tonemapped+no_extra_gaussians_with_densif_pruning --num_init_points 0


bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/tonemapped+init_pruning_with_densif_pruning


bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/tonemapped+init_pruning++no_densif_with_densif_pruning --densify_from_iter 999999999999


bash run.sh --raytracer_version v15_1bounce_noblur_ignore0vol -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp54/ignore0vol

