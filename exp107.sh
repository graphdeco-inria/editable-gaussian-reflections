


python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp107/baseline_force0 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 2000 --densif_no_splitting --lod_force_blur_sigma 0.0

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/no_split --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 2000 --densif_no_splitting 

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/no_clone --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 2000 --densif_no_cloning 

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/baseline_force0_nominsize --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 2000 --no_lod_clamp_minsize


# 

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/no_split_power_1_withoptim --use_diffuse_target  --densify_from_iter 2000 --densif_no_splitting 