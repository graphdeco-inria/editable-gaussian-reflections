

build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_${BASH_SOURCE[0]/.sh/}/${build}_no_densify --densify_from_iter 99999999999


build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_${BASH_SOURCE[0]/.sh/}/${build}_no_densify_add_noise --densify_from_iter 99999999999 --add_mcmc_noise

build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_${BASH_SOURCE[0]/.sh/}/${build}_add_noise --add_mcmc_noise



build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom --glossy_loss_weight 0.0 -m output_${BASH_SOURCE[0]/.sh/}/${build}_no_extra_points --num_farfield_init_points 0 
