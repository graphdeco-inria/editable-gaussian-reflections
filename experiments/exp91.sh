
q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp91/baseline --init_lod_prob_0 1.0 --lod_init_scale 0.002 --lod_clamp_minsize --prob_blur_targets 1.0 --densify_from_iter 999999"

# 

q a40 <<< "bash run.sh --raytracer_version v23_t0.001_localsort -s colmap/shiny_bedroom -m output_exp91/localsort --init_lod_prob_0 1.0 --lod_init_scale 0.002 --lod_clamp_minsize --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001_localsort_nolod -s colmap/shiny_bedroom -m output_exp91/localsort_nolod --init_lod_prob_0 1.0 --lod_clamp_minsize --lod_init_scale 0.002 --prob_blur_targets 1.0 --densify_from_iter 999999"

# 

q a40 <<< "bash run.sh --raytracer_version v23_t0.05 -s colmap/shiny_bedroom -m output_exp91/localsort --init_lod_prob_0 1.0 --lod_init_scale 0.002 --lod_clamp_minsize --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "bash run.sh --raytracer_version v23_t0.05_localsort -s colmap/shiny_bedroom -m output_exp91/localsort --init_lod_prob_0 1.0 --lod_init_scale 0.002 --lod_clamp_minsize --prob_blur_targets 1.0 --densify_from_iter 999999s"
