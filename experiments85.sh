
# is the low quality due to raytracer settings, sort order or gaussian thresholding?

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp85a/baseline --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort -s colmap/shiny_bedroom -m output_exp85a/globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort_t0.001 -s colmap/shiny_bedroom -m output_exp85a/globalsort_t0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce_globalsort_t0.001_a0.001 -s colmap/shiny_bedroom -m output_exp85a/globalsort_t0.001_a0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"


# --

# is the low quality affects by the initialization lod scale?

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp85b/init_0.01 --init_lod_prob_0 1.0 --init_lod_scale 0.01 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp85b/init_0.05 --init_lod_prob_0 1.0 --init_lod_scale 0.05 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp85b/init_0.1 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp85b/init_0.5 --init_lod_prob_0 1.0 --init_lod_scale 0.5 --prob_blur_targets 1.0 --densify_from_iter 999999"

# --

# is the low quality affects by the initialization being empty is higher lods? if we fill them in at init, is the result better?

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v21_lod_0bounce_globalsort -s colmap/shiny_bedroom -m output_exp84/init_dupe_globalsort --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v21_lod_0bounce -s colmap/shiny_bedroom -m output_exp84/init_dupe_baseline --init_lod_prob_0 1.0 --init_lod_scale 0.1 --prob_blur_targets 1.0 --densify_from_iter 999999"
