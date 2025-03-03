



q a40 <<< "bash run.sh --raytracer_version v23 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.002 --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.01 --init_lod_prob_0 1.0 --lod_init_scale 0.01 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.04 --init_lod_prob_0 1.0 --lod_init_scale 0.04 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

# --------

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.002_t0.001 --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.01_t0.001 --init_lod_prob_0 1.0 --lod_init_scale 0.01 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp87/init_scale_0.04_t0.001 --init_lod_prob_0 1.0 --lod_init_scale 0.04 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

