

NO_TONEMAPPING=1 bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp55/fixed_baseline --no_znear_init_pruning

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp55/fixed_tonemapped --no_znear_init_pruning

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp55/fixed_tonemapped+znear


bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp55/fixed_tonemapped+znear_during_densif --znear_densif_pruning --max_images 2

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp55/fixed_tonemapped+znear+nodensif --densify_from_iter 999999999999