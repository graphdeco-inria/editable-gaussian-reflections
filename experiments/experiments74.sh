




q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0 --init_lod_prob_0 1.0"
q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_noscale --lod_mean_lr 0.0  --init_lod_prob_0 1.0"
q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_nomean --lod_scale_lr 0.0 --init_lod_prob_0 1.0"


q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.5 --init_lod_prob_0 1.0 --lod_init_scale 0.5"
q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2 --init_lod_prob_0 1.0 --lod_init_scale 0.2"
q a6000 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.05 --init_lod_prob_0 1.0 --lod_init_scale 0.05"

