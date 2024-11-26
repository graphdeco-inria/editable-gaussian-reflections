
# Why is it better sometimes?

q a40 -n offset_0.0_sanity_check <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --brdf_mode static_lut --use_attached_brdf"
q a40 -n offset_0.0_r2 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset 0.0 --brdf_mode static_lut --use_attached_brdf"
q a40 -n offset_0.0_r3 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset 0.0 --brdf_mode static_lut --use_attached_brdf"
q a40 -n offset_0.0_r4 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset 0.0 --brdf_mode static_lut --use_attached_brdf"
q a40 -n offset_0.0_r5 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset 0.0 --brdf_mode static_lut --use_attached_brdf"