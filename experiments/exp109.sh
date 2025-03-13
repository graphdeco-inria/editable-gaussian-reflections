q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/baseline --use_diffuse_target  --densif_no_splitting"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/no_scale_lr --use_diffuse_target  --densif_no_splitting --lod_scale_lr 0.0"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/init_scale_halved --use_diffuse_target  --densif_no_splitting --lod_init_scale 0.0025"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/init_scale_halved_no_scale_lr --use_diffuse_target  --densif_no_splitting --lod_init_scale 0.0025 --lod_scale_lr 0.0"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/power2 --use_diffuse_target  --densif_no_splitting --lod_schedule_power 2"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/from2k --use_diffuse_target  --densif_no_splitting   --densify_from_iter 2000"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/from6k --use_diffuse_target  --densif_no_splitting   --densify_from_iter 6000"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/from12k --use_diffuse_target  --densif_no_splitting   --densify_from_iter 12000"

q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/from12k_power2 --use_diffuse_target  --densif_no_splitting   --densify_from_iter 12000 --lod_schedule_power 2"






# q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp108/baseline_with_splitting --use_diffuse_target"



python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp109/baseline_with_splitting_fix --use_diffuse_target