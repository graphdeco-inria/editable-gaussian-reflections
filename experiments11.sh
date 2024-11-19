


q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/gt_brdf_sanity_check_compute_ray_code --brdf_mode finetuned_lut --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/gt_brdf_static --brdf_mode static_lut --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/baseline_static --brdf_mode static_lut --use_attached_brdf --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/detach_position_static --brdf_mode static_lut --use_attached_brdf --detach_position --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/detach_normal_static --brdf_mode static_lut --use_attached_brdf --detach_normal --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/gt_brdf_sanity_check_compute_ray_code_static --brdf_mode static_lut --iterations 90000"