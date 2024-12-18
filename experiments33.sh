


# done, works
# REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-bg-stochastic -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32_nodens_but_mcmcinit_refl0  --brdf_mode static_lut --force_mcmc_custom_init

# done, fails without the change to the init
# REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-bg-stochastic -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32_nodens_refl0  --brdf_mode static_lut; 

# done, breaks when densification kicks in (I assume since doubling the # of gaussians while putting them one atop the other increases the total # of hits)
# REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-bg-stochastic -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32_refl0  --brdf_mode static_lut --mcmc_densify --cap_max 400000  

# with alpha threshold 0.05, works fine with regular init, but still fails with densification
# REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-stochastic-alphat-0.05 -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32_nodens_refl0_alphat0.05  --brdf_mode static_lut; REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-stochastic-alphat-0.05 -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32_refl0_alphat0.05  --brdf_mode static_lut --mcmc_densify --cap_max 400000  


REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-bg-normalized-alphat-0.05 -s colmap/hard_kitchen_mirror/ -m output_max16/max16+norm32_nodens_refl0_alphat0.05  --brdf_mode static_lut;  REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build-fill-bg-normalized-alphat-0.05 -s colmap/hard_kitchen_mirror/ -m output_max16/max16+norm32_refl0_alphat0.05  --brdf_mode static_lut --mcmc_densify --cap_max 400000  

# ---------------------------------------------


REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build-fill-bg-stochastic -s colmap/hard_kitchen_mirror/ -m output_max16/max16+stoch32 --mcmc_densify --cap_max 400000 --brdf_mode static_lut

REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.5 -s colmap/hard_kitchen_mirror/ -m output_max16/baseline_v0.5 --mcmc_densify --cap_max 400000 --brdf_mode static_lut

REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.5 -s colmap/hard_kitchen_mirror/ -m output_max16/baseline_v0.5_nodens --brdf_mode static_lut

REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build_v0.5 -s colmap/hard_kitchen_mirror/ -m output_max16/baseline_v0.5_nodens_refl0 --brdf_mode static_lut