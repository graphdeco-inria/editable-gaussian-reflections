

for build in v6_baseline v6_normalize_normal_map \
v6_project_position_to_ray \
v6_project_and_normalize \
v6_2ndbounce_start_5k \
v6_2ndbounce_start_10k \
v6_2ndbounce_start_15k \
v6_2ndbounce_start_20k \
v6_2ndbounce_start_25k \
v6_init_f0_0.04 \
v6_init_f0_0.0 \
v6_valid_normal_0.1 \
v6_valid_normal_0.2 \
v6_valid_normal_0.4 \
v6_valid_normal_0.8 \
v6_unweighted_rgb; do 
q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version $build -s colmap/hard_kitchen_mirror -m output_exp11/$build"
done