# Try different configs

for path in raytracer_builds/build_v0.6*; do 
    q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_v0.6_try2/$(basename $path)"
done

for path in raytracer_builds/build_v0.6_a0.01*; do 
    q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version $(basename $path) -s colmap/hard_kitchen_mirror -m output_v0.6_try2/$(basename $path)_with_opacity_resets --use_opacity_resets"
done


