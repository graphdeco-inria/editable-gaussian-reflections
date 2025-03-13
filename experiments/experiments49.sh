
build=v15_0bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_jitter_clones --densif_jitter_clones

build=v15_0bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_scaledown_clones --densif_scaledown_clones

# --

build=v15_0bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_jitter_and_scaledown_clones --densif_jitter_clones --densif_scaledown_clones

build=v15_0bounce_noblur_jitter; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_jitterray

build=v15_0bounce_noblur_allhits; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_allhits

build=v15_0bounce_noblur; bash run.sh --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp49/${build}_tonemapped0bounce_minsize_0.0001 --min_gaussian_size 0.0001