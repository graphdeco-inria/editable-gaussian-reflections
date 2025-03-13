
# todo: fix err below
# for path in raytracer_builds/build-v0.5-*; do q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build_fillincolor_alphat0.01 -s colmap/hard_kitchen_mirror -m output_detached/$name --mcmc_densify --mcmc_skip_relocate --cap_max 600000"; done

# todo: baed ont hese changes:
# path=raytracer_builds/build-v0.5-detachafter64-alphat0.05; REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_detached/$(basename $path)_fulldensif --mcmc_densify --cap_max 600000
