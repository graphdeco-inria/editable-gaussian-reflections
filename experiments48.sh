
build=v15_1bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp48/${build}_tonemapped_no_splitting --densif_no_splitting

build=v15_1bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp48/${build}_tonemapped_no_cloning --densif_no_cloning


build=v15_1bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp48/${build}_tonemapped_znear --znear_init_pruning

build=v15_1bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp48/${build}_tonemapped_pruning_only --densif_pruning_only