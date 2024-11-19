
# 13) still getting poor results with my ray computation & g.t. brdf, try to use the precomputed ray again 

q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v3/gt_brdf_precomp_ray --brdf_mode static_lut --precomp_ray"
q a6000 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_correct_reflray_v3/gt_brdf_static --brdf_mode static_lut --precomp_ray"


