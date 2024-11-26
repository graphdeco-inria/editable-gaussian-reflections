#### Try MCMC densification in different configurations

## 1) Try it without any reflections to see how good the diffuse gets

q a40 -n no_refl_no_dens <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_norefl/baseline"

q a40 -n no_refl_with_dens <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_norefl/400k  --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

## 2) Try different weights of their regularizer

q a40 -n 400k_realray_reg0.0 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.0 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

q a40 -n 400k_realray_reg0.0001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.0001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0001 --scale_reg 0.0001 --cap_max 400000"

q a40 -n 400k_realray_reg0.001 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.001 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.001 --scale_reg 0.001 --cap_max 400000"

q a40 -n 400k_realray_reg0.01 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_realray_reg/0.01 --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.01 --scale_reg 0.01 --cap_max 400000"

## 3) Does it help with when we are using real reflection rays, why caused poor fitting of the normals & ugly reflections?

q a40 -n baseline_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/baseline --brdf_mode static_lut --use_attached_brdf"

q a40 -n 400k_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/400k --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

q a40 -n 800k_realray <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_real_rays/800k --brdf_mode static_lut --use_attached_brdf --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 800000"

## 4) Does it fix the need to reduce the weight on the reflection loss?

q a40 -n baseline_refl1.0 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl1.0/baseline"

q a40 -n 400k_refl1.0 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl1.0/400k --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

q a40 -n 800k_refl1.0 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl1.0/800k --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 800000"

# --

q a40 -n baseline_refl0.1 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl0.1/baseline"

q a40 -n 400k_refl0.1 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl0.1/400k --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 400000"

q a40 -n 800k_refl0.1 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_noreg_refl0.1/800k --mcmc_densify --opacity_reg 0.0 --scale_reg 0.0 --cap_max 800000"

