


path=raytracer_builds/build; q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_tthresh/tthresh_0.03_alphat0.5_respawn_only --mcmc_densify"

path=raytracer_builds/build; q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_tthresh/tthresh_0.03_alphat0.5_400k --mcmc_densify --cap_max 400000"

path=raytracer_builds/build; q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_tthresh/tthresh_0.03_alphat0.5_nodens"