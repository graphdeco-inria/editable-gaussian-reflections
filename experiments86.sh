

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp84/v22_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort_a0.001 -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_a0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort_t0.001 -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_t0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "SKIP_CLAMP_MINSIZE=1 DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_noclamp --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_slowdown --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999 --slowdown 2"


q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_init_0.01 --init_lod_prob_0 1.0 --init_lod_scale 0.01 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp84/v22_globalsort_init_0.05 --init_lod_prob_0 1.0 --init_lod_scale 0.05 --prob_blur_targets 1.0 --densify_from_iter 999999"