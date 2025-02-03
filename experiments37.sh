# Try different configs

# for path in raytracer_builds/{build_v0.6_a0.05_t0.1,build_v0.6_a0.01_t0.001}; do 
# for path in raytracer_builds/build_v0.6_a0.01_t0.1; do 
#     q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_v0.6_try2/$(basename $path)_densify_400k --mcmc_densify --cap_max 400000"
# done

# for path in raytracer_builds/build_v0.6_a0.01_t0.1; do 
#     q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_v0.6_try2/$(basename $path)_densify_without_increase --mcmc_densify"
# done


for path in raytracer_builds/{build_v0.6_a0.05_t0.1,build_v0.6_a0.01_t0.001,build_v0.6_a0.01_t0.1}; do 
    q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_v0.6_try2/$(basename $path)_densify_400k_with_resets --mcmc_densify --cap_max 400000 --use_opacity_resets"
done