
## 1) Review the current state of all experiments to make sure I didn't introduce bugs


# q a40 -n current_baseline_v0.1 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/baseline_v0.1"

q a40 -n current_baseline_v0.2 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4 -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/baseline_v0.4"

q a40 -n current_mcmc <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4 -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/mcmc_v0.4 --mcmc_densify --cap_max 400000"

q a40 -n current_lut <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4 -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/lut_v0.4 --brdf_mode static_lut"

q a40 -n current_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4 -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/lut_attached_v0.4 --brdf_mode static_lut --use_attached_brdf"


q a40 -n current_everything_combined <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.4 -s colmap/hard_kitchen_mirror/ -m output_current_state_v0.4/everything --mcmc_densify --cap_max 400000 --brdf_mode static_lut --use_attached_brdf"
