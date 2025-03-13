
build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp44/${build}_with_custom_init_500k

build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 --mcmc_densify_disable_custom_init -m output_exp44/${build}_no_custom_init_500k

build=v15_0bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_exp44/${build}_baseline_0bounce_with_custom_init_500k

build=v15_0bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 --mcmc_densify_disable_custom_init -m output_exp44/${build}_no_custom_init_0bounce_500k


