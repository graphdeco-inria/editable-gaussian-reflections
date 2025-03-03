
q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp96/no_densif --lod_init_scale 0.005 --prob_blur_targets 1.0 --densify_from_iter 999999"


# Does the "wobbling" when switching lod level go away if the lod scale is fixed to a constant?
q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp96/no_densif_lod_lr_0.0 --lod_init_scale 0.005 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"


q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp96/no_densif_5x_lod_scale_lr --lod_init_scale 0.005 --prob_blur_targets 1.0 --densify_from_iter 999999  --lod_scale_lr 0.00025"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp96/no_densif_25x_lod_scale_lr --lod_init_scale 0.005 --prob_blur_targets 1.0 --densify_from_iter 999999  --lod_scale_lr 0.00125"

q a40 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp96/no_densif_125x_lod_scale_lr --lod_init_scale 0.005 --prob_blur_targets 1.0 --densify_from_iter 999999  --lod_scale_lr 0.00625"



