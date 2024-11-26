
## 3) Does it help with when we are using real reflection rays, why caused poor fitting of the normals & ugly reflections?

q a40 -n baseline_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/baseline --brdf_mode static_lut --use_attached_brdf"

q a40 -n 400k_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/400k --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

q a40 -n 800k_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/800k --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 800000"


## 1) Try very low weights of their regularizer

q a40 -n 400k_realray_reg00000001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/00000001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.00000001 --scale_reg 0.00000001 --cap_max 400000"

q a40 -n 400k_realray_reg0000001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0000001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0000001 --scale_reg 0.0000001 --cap_max 400000"

q a40 -n 400k_realray_reg0.000001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.000001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.000001 --scale_reg 0.000001 --cap_max 400000"

q a40 -n 400k_realray_reg0.00001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.0001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.00001 --scale_reg 0.00001 --cap_max 400000"

q a40 -n 400k_realray_reg0.0001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.0001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0001 --scale_reg 0.0001 --cap_max 400000"

q a40 -n 400k_realray_reg0.001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0001 --scale_reg 0.001 --cap_max 400000"

q a40 -n 400k_realray_reg0.01 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.01 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0001 --scale_reg 0.01 --cap_max 400000"