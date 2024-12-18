
## relaunch on old raytracer to make sure there is no quality regression

q a40 -n current_baseline <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/baseline"

# q a40 -n current_mcmc <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/mcmc --mcmc_densify --cap_max 400000"

# q a40 -n current_mcmc+lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/mcmc+lut --mcmc_densify --cap_max 400000 --brdf_mode static_lut"


# # ---------------

q a40 -n current_lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/lut --brdf_mode static_lut"

q a40 -n current_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/lut_attached --brdf_mode static_lut --use_attached_brdf"

# q a40 -n current_everything <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.1/everything --mcmc_densify --cap_max 400000 --brdf_mode static_lut --use_attached_brdf"