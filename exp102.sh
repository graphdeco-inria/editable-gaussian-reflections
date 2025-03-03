python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_plus_1e-5 --use_diffuse_target --lod_force_blur_sigma 0.0


python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_min_1e-8 --use_diffuse_target --lod_force_blur_sigma 0.0






SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_skipclamp --use_diffuse_target --lod_force_blur_sigma 0.0

# 


SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_nolr --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0

SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_nolr_force0 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_force_blur_sigma 0.0

SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_nomeanlr --use_diffuse_target --lod_mean_lr 0.0

SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_noscalelr --use_diffuse_target --lod_scale_lr 0.0


# 

SKIP_CLAMP_MINSIZE=1 python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp102/noblur_atleast_1e-8_noscalelr --use_diffuse_target --lod_scale_lr 0.0 --lod_force_blur_sigma 0.0

