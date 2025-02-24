

q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.01 -m output_exp73/lod"


q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.01 -m output_exp73/lod_noopt --lod_mean_lr 0.0 --lod_scale_lr 0.0"

q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.01 -m output_exp73/lod_noopt_init0 --lod_mean_lr 0.0 --lod_scale_lr 0.0 --init_lod_prob_0 1.0"

q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.01 -m output_exp73/lod_noopt_nodensif --lod_mean_lr 0.0 --lod_scale_lr 0.0 --densify_from_iter 9999999"
