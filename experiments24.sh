


## 1) Try very low weights of their regularizer with the ground turth brdf and with a higher cap 

q a40 -n 400k_realray_reg1e-12 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/00000001 --mcmc_densify --opacity_reg 1e-10 --scale_reg 1e-10 --cap_max 400000"

q a40 -n 400k_realray_reg1e-10 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/00000001 --mcmc_densify --opacity_reg 1e-10 --scale_reg 1e-10 --cap_max 400000"

q a40 -n 400k_realray_reg1e-8 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/00000001 --mcmc_densify --opacity_reg 1e-8 --scale_reg 1e-8 --cap_max 400000"

q a40 -n 400k_realray_reg1e-6 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/1e-6 --mcmc_densify --opacity_reg 1e-6 --scale_reg 1e-6 --cap_max 400000"

q a40 -n 400k_realray_reg1e-4 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/1e-4 --mcmc_densify --opacity_reg 1e-4 --scale_reg 1e-6 --cap_max 400000"

q a40 -n 400k_realray_reg1e-2 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify_gt_brdf_fixthresh/1e-2 --mcmc_densify --opacity_reg 1e-2 --scale_reg 1e-6 --cap_max 400000"
