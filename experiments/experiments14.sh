
# 12) output_correct_reflray_v4

# Previous run had bugs (incorrect camera transpose). First re-run with g.t. brdf as a sanity check, to make sure that the code is correct. (should obtain good quality, matching previous results). 

q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_detached_fixed/gt_brdf_baseline --brdf_mode gt"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_detached_fixed/gt_refl_ray_static --brdf_mode static_lut"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_detached_fixed/gt_refl_ray_finetuned --brdf_mode finetuned_lut"
