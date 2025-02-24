
q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort_t0.001 -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort_t0.001_a0.001 -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"


# --

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.01_nodensif --init_lod_prob_0 1.0 --init_lod_scale 0.01 --prob_blur_targets 0.5 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif_prob_blur_1.0 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

# --

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v21_lod_0bounce_globalsort_t0.001 -s colmap/shiny_bedroom -m output_exp84/optimized_init_0.1_nodensif_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 0.5 --densify_from_iter 999999"
