
## 1) Review the current state of all experiments to make sure I didn't introduce bugs


# q a40 -n current_baseline <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/baseline"

# q a40 -n current_mcmc <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/mcmc --mcmc_densify --cap_max 400000"

# q a40 -n current_mcmc+lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/mcmc+lut --mcmc_densify --cap_max 400000 --brdf_mode static_lut"


# # ---------------

# q a40 -n current_lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/lut --brdf_mode static_lut"

# q a40 -n current_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/lut_attached --brdf_mode static_lut --use_attached_brdf"

# q a40 -n current_everything <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/everything --mcmc_densify --cap_max 400000 --brdf_mode static_lut --use_attached_brdf"


# --------------

## Relaucnh experiments with mistakes in file name output

q a40 -n current_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/lut_attached --brdf_mode static_lut --use_attached_brdf"

q a40 -n current_attached+rayoffset0 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/lut_attached_ray_offset --brdf_mode static_lut --use_attached_brdf --ray_offset 0.0"

q a40 -n current_attached+rayoffset0.01 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/lut_attached_ray_offset0.01 --brdf_mode static_lut --use_attached_brdf --ray_offset 0.01"

q a40 -n attached_oldbuild_v0.1 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/attached_oldbuild_v0.1 --brdf_mode static_lut --use_attached_brdf"


## Current experiments with mcmc are crashing... try a different opacity pruning threshold

q a40 -n current_mcmc <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/mcmc_bighresh --mcmc_densify --cap_max 400000 --opacity_pruning_threshold 0.015"

q a40 -n current_mcmc+lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/mcmc_bighresh+lut --mcmc_densify --cap_max 400000 --brdf_mode static_lut --opacity_pruning_threshold 0.015"

q a40 -n current_everything_combined <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4b -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4b/everything_bighresh --mcmc_densify --cap_max 400000 --brdf_mode static_lut --use_attached_brdf --opacity_pruning_threshold 0.015"
