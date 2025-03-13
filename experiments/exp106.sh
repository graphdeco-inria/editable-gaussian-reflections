


python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/densify_from_2k --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 2000

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/densify_from_4k --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 4000

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/densify_from_8k --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 8000

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp106/densify_from_16k --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.5 --densify_from_iter 16000

