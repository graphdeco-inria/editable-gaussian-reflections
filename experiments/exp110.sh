q a40 <<< "python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp110/with_splitting_interval_500  --use_diffuse_target --densify_from_iter 1500 --densification_interval 500"

q a40 <<< "python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp110/with_splitting_interval_1000  --use_diffuse_target --densify_from_iter 1500 --densification_interval 1000"

q a40 <<< "python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp110/with_splitting_interval_2000  --use_diffuse_target --densify_from_iter 1500 --densification_interval 2000"

# 

q a40 <<< "ON_SPLIT_SKIP_DONT_TOUCH_LOD=1 python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp110/dont_touch --use_diffuse_target --densify_from_iter 1500 --densification_interval 500"

