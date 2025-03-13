



q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/gt_brdf--brdf_mode finetuned_lut --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/baseline--brdf_mode finetuned_lut --use_attached_brdf --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/detach_position--brdf_mode finetuned_lut --use_attached_brdf --detach_position --iterations 90000"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray/detach_normal--brdf_mode finetuned_lut --use_attached_brdf --detach_normal --iterations 90000"
