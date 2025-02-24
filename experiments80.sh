

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_frozen_scale_0.2 --init_lod_prob_0 1.0 --lod_scale_lr 0.0 --lod_mean_lr 0.0 --init_lod_scale 0.2"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_frozen_scale_1.0 --init_lod_prob_0 1.0 --lod_scale_lr 0.0 --lod_mean_lr 0.0 --init_lod_scale 1.0"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_frozen_scale_1.0 --init_lod_prob_0 1.0 --lod_scale_lr 0.0 --lod_mean_lr 0.0 --init_lod_scale 5.0"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_frozen_scale_9999.0 --init_lod_prob_0 1.0 --lod_scale_lr 0.0 --lod_mean_lr 0.0 --init_lod_scale 9999.0"



q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_unfreeze_mean_amp_1.00 --init_lod_prob_0 1.0 --lod_mean_lr 0.0 --init_lod_scale 1.0 --lod_scale_decay 1.00"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_unfreeze_mean_amp_1.001 --init_lod_prob_0 1.0 --lod_mean_lr 0.0 --init_lod_scale 1.0 --lod_scale_decay 1.001"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp80/lod_0bounce_unfreeze_mean_amp_1.01 --init_lod_prob_0 1.0 --lod_mean_lr 0.0 --init_lod_scale 1.0 --lod_scale_decay 1.01"




# ----------------------------------------
