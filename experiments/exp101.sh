 
 
 
python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp109/with_splitting --use_diffuse_target

LOD_CLAMP_EPS=0.0 python3 train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp109/without_splitting_without_eps --use_diffuse_target --densif_no_splitting