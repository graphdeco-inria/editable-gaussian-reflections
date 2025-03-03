


q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/densif --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/prob_blur_50% --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.5 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/lod_init_scale_0.0001 --init_lod_prob_0 1.0 --lod_init_scale 0.0001 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/lod_init_scale_0.0005 --init_lod_prob_0 1.0 --lod_init_scale 0.0005 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/slowdown2 --init_lod_prob_0 1.0 --lod_init_scale 0.002 --slowdown 2 --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/slowdown2_prob_blur_50% --init_lod_prob_0 1.0 --lod_init_scale 0.002 --slowdown 2 --prob_blur_targets 0.5 --densify_from_iter 999999 --lod_clamp_minsize"


q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0% --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 --densify_from_iter 999999 --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/scale_decay_0.9999 --init_lod_prob_0 1.0 --lod_init_scale 0.002 --scale_decay 0.9999 --prob_blur_targets 1.0  --densify_from_iter 999999 --lod_clamp_minsize"

# --

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0%_with_densif --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0  --lod_clamp_minsize"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0%_with_densif_noclamp --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 "

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0%_noclamp --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 --densify_from_iter 999999"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0%_nolod_lr --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 --lod_mean_lr 0.0 --lod_scale_lr 0.0 --densify_from_iter 999999"


q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/sanity_check_prob_blur_0%_nolod_lr_withclamp --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 --lod_mean_lr 0.0 --lod_scale_lr 0.0 --densify_from_iter 999999 --lod_clamp_minsize"


q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/10xlr --densify_from_iter 999999  --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --lod_clamp_minsize --lod_mean_lr 0.0005 --lod_scale_lr 0.0005"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp89/0.1xlr --densify_from_iter 999999  --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --lod_clamp_minsize --lod_mean_lr 0.000005 --lod_scale_lr 0.000005"

# 


q a40 <<< "bash run.sh --raytracer_version v20_nolod_0bounce -s colmap/shiny_bedroom -m output_exp89/nolod --densify_from_iter 999999  --init_lod_prob_0 1.0 --lod_init_scale 0.002  --lod_clamp_minsize "

q a40 <<< "bash run.sh --raytracer_version v20_nolod_0bounce -s colmap/shiny_bedroom -m output_exp89/nolod_noclamp --densify_from_iter 999999  --init_lod_prob_0 1.0 --lod_init_scale 0.002 "

q a40 <<< "bash run.sh --raytracer_version v20_nolod_0bounce -s colmap/shiny_bedroom -m output_exp89/nolod_withdensif  --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 1.0 --lod_clamp_minsize"

# todo: no lod at all, + sweep 


# 

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp90/sanity_check_prob_blur_0%_with_densif_noclamp --init_lod_prob_0 1.0 --lod_init_scale 0.002 --prob_blur_targets 0.0 "
