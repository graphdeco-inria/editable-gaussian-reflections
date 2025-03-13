


python train.py --raytracer_version v24 -s colmap/shiny_bedroom -m output_exp104/v24 --use_diffuse_target --lod_scale_lr 0.0 --lod_force_blur_sigma 0.0


python train.py --raytracer_version v24 -s colmap/shiny_bedroom -m output_exp104/v24 --use_diffuse_target --lod_scale_lr 0.0 --lod_force_blur_sigma 0.0


# 


DISABLE_LOD_INIT=1 python train.py --raytracer_version v24 -s colmap/shiny_bedroom -m output_exp104/v24_disableinit --use_diffuse_target --lod_scale_lr 0.0 --lod_force_blur_sigma 0.0


DISABLE_LOD_INIT=1 python train.py --raytracer_version v24_noflowscaletolodmean -s colmap/shiny_bedroom -m output_exp104/v24_noflowscaletolodmean_disableinit --use_diffuse_target --lod_scale_lr 0.0 --lod_force_blur_sigma 0.0

