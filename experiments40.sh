
// v7_baseline                // normalizing normal + init f0 set to 0. init f0 didnt work as much as I would have liked, probably because of interpenetration
// v7_project_position_to_ray // normalizing normal + project posotion worked, but project position alone failed. need to try again

// v7_skip_backface_0.0
// v7_skip_backface_0.3
// v7_skip_backface_0.6

// v7_unweighted_rgb


# ---------------------------------------------------------


build=v7_baseline; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build
build=v7_project_position_to_ray; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build


build=v7_skip_backface_0.0; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build
build=v7_skip_backface_0.3; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build
build=v7_skip_backface_0.6; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build

build=v7_unweighted_rgb; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build

build=v7_valid_normal_0.3; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build
build=v7_valid_normal_0.6; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp12/$build


# -------------------

build=v8_baseline; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build


build=v8_valid_0.3f; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build

build=v8_valid_0.6f; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build

build=v8_valid_0.9f; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build

build=v8_delay_20k; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build


build=v8_save_lut_imagesf; GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp13/$build