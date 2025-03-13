# Goal: fix black spots

# try g.t. individual passes 
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_black_spots_with_offsets/detached_normals --brdf_mode static_lut --use_attached_brdf --detach_normals"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_black_spots_with_offsets/detached_F0 --brdf_mode static_lut --use_attached_brdf --detach_F0"

# A. try different ray offsets
for OFFSET in 0.0 0.001 0.01 0.02 0.04 0.08; do 
    q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_black_spots_with_offsets/$OFFSET --brdf_mode static_lut --use_attached_brdf --ray_offset $OFFSET"
done;

# REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_black_spots_with_offsets/attached --brdf_mode static_lut --use_attached_brdf --detach_position

# C. try warmup 
