


q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_0.1 --init_lod_prob_0 1.0 --lod_init_scale 0.1 --prob_blur_targets 0.5"
q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_1.0 --init_lod_prob_0 1.0 --lod_init_scale 1.0 --prob_blur_targets 0.5"
q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_10.0 --init_lod_prob_0 1.0 --lod_init_scale 10.0 --prob_blur_targets 0.5"


# ----------------------------------------


q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_0.1_nodensif --init_lod_prob_0 1.0 --lod_init_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"
q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_1.0_nodensif --init_lod_prob_0 1.0 --lod_init_scale 1.0 --prob_blur_targets 0.5 --densify_from_iter 999999"
q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp83/optimized_init_10.0_nodensif --init_lod_prob_0 1.0 --lod_init_scale 10.0 --prob_blur_targets 0.5 --densify_from_iter 999999"

