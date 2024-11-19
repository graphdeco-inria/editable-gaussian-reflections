


# Then, run with the attached brdf, trying both static and finetuned lut & with different parameters detached to isolate where the errors come from

q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/baseline_static --brdf_mode static_lut --use_attached_brdf"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/baseline_finetuned --brdf_mode finetuned_lut --use_attached_brdf"

q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/detach_position_static --brdf_mode static_lut --use_attached_brdf --detach_position"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/detach_normal_static --brdf_mode static_lut --use_attached_brdf --detach_normal"

q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/detach_position_finetuned --brdf_mode finetuned_lut --use_attached_brdf --detach_position"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_attached_fixed/detach_normal_finetuned --brdf_mode finetuned_lut --use_attached_brdf --detach_normal"