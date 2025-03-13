

bash run.sh --raytracer_version v15_1bounce_noblur -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp53/1bounce_noblur

bash run.sh --raytracer_version v15_1bounce -s colmap/shiny_bedroom --glossy_loss_weight 0.000 --blur_kernel_bandwidth 0.2 --init_blur_level_prob_0 0.5 -m output_exp53/0weight_blur

bash run.sh --raytracer_version v15_1bounce_no_estimation -s colmap/shiny_bedroom --glossy_loss_weight 0.000 --blur_kernel_bandwidth 0.2 --init_blur_level_prob_0 0.5 -m output_exp53/0weight_blur_noestimation

