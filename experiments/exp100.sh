


for level in 0.001 0.01 0.05 0.1 0.2 0.3; do 
    q a40 <<< "DONT_DENSIFY_HIGH_LOD=$level python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp100/dont_densify_high_lod_$level"
done

