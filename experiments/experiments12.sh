
# 12) output_correct_reflray_v2
# Previous run had bugs (incorrect camera transpose). First re-run with g.t. brdf as a sanity check, to make sure that the code is correct. (should obtain good quality, matching previous results). 

q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/gt_brdf_sanity_check_compute_ray_code --brdf_mode finetuned_lut"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/gt_brdf_sanity_check_compute_ray_code_static --brdf_mode static_lut"

# Then, run with the attached brdf, trying both static and finetuned lut & with different parameters detached to isolate where the errors come from

q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/baseline_static --brdf_mode static_lut --use_attached_brdf"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/baseline_finetuned --brdf_mode finetuned_lut --use_attached_brdf"

q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/detach_position_static --brdf_mode static_lut --use_attached_brdf --detach_position"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/detach_normal_static --brdf_mode static_lut --use_attached_brdf --detach_normal"

q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/detach_position_finetuned --brdf_mode finetuned_lut --use_attached_brdf --detach_position"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v2/detach_normal_finetuned --brdf_mode finetuned_lut --use_attached_brdf --detach_normal"