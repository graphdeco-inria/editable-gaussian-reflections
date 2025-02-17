

build=v15_1bounce_noblur; CLAMP28=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_area --glossy_loss_weight 0.0

# -----------------------------------------------------------------

build=v15_1bounce_noblur; CLAMP28=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_clamp28 --glossy_loss_weight 0.0

build=v15_1bounce_noblur; CLAMP21=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_clamp21 --glossy_loss_weight 0.0

build=v15_1bounce_noblur; CLAMP51=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_clamp51 --glossy_loss_weight 0.0

build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_losses0 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --specular_loss_weight 0.0 --albedo_loss_weight 0.0 --metalness_loss_weight 0.0

# -----------------------------------------------------------------

build=v15_1bounce_noblur; CLAMP01=1 python3 train.py --raytracer_version $build -s colmap/shiny_bedroom -m output_exp43/${build}_no_f0 --glossy_loss_weight 0.0 --f0_loss_weight 0.0 

