


python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp105/power_1 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 1.0

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp105/power_2 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 2.0

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp105/power_4 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 4.0

python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp105/power_8 --use_diffuse_target --lod_scale_lr 0.0 --lod_mean_lr 0.0 --lod_schedule_power 8.0
