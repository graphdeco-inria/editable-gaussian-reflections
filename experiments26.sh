
## 1) Is there a loss in quality attributable to their bad init?
q a40 -n mcmc_their_init <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_init/their_init --mcmc_densify --cap_max 400000"

q a40 -n mcmc_our_init <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_init/our_init --mcmc_densify --cap_max 400000 --mcmc_densify_disable_custom_init"

## 2) Try the scaling reg alone, without the opacity reg
for x in 1e-12 1e-8 1e-4 1e-2 1e-1; do 
    q a40 -n scale_reg$x <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_scale_reg/scale_reg$x --mcmc_densify  --scale_reg $x --cap_max 400000"
done

## 3) Try the opacotu reg alone, without the scaling loss
for x in 1e-12 1e-8 1e-4 1e-2 1e-1; do 
    q a40 -n opacity_reg$x <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.2_mcmc -s colmap/hard_kitchen_mirror/ -m output_mcmc_opacity_reg/opacity_reg$x --mcmc_densify --opacity_reg $x --cap_max 400000"
done
