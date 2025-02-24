


q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/with_densif --init_lod_prob_0 1.0 --init_lod_scale 0.1 --lod_scale_lr 0.0 --prob_blur_targets 1.0"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/scalelr0_init0.1_with_densif --init_lod_prob_0 1.0 --init_lod_scale 0.1 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"


q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/scalelr0_init0.05 --init_lod_prob_0 1.0 --init_lod_scale 0.05 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/scalelr0_init0.1 --init_lod_prob_0 1.0 --init_lod_scale 0.1 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/scalelr0_init0.2 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"

q a40 <<< "DUPLICATE_INIT_PCD=1 python train.py --raytracer_version v22_globalsort -s colmap/shiny_bedroom -m output_exp65/scalelr0_init0.4 --init_lod_prob_0 1.0 --init_lod_scale 0.4 --lod_scale_lr 0.0 --prob_blur_targets 1.0 --densify_from_iter 999999"