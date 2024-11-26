## Densification


q a40 -n mcmc_densify <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_mcmc_densify/mcmc_densify --mcmc_densify"
