
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.5_noopt --init_lod_prob_0 1.0 --init_lod_scale 0.5  --lod_mean_lr 0.0  --lod_scale_lr 0.0 --init_lod_prob_0 1.0"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noopt --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_mean_lr 0.0  --lod_scale_lr 0.0"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.05_noopt --init_lod_prob_0 1.0 --init_lod_scale 0.05 --lod_mean_lr 0.0  --lod_scale_lr 0.0"
#!!!!!!!!!!! init_lod_scale does not appear to be respected

q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_opacity_reg_0.1 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --opacity_reg 0.1"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_opacity_reg_0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --opacity_reg 0.001"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_opacity_reg_0.00001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --opacity_reg 0.00001"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_opacity_reg_0.0000001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --opacity_reg 0.0000001"

q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_scale_reg_0.1 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --scale_reg 0.1"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_scale_reg_0.001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --scale_reg 0.001"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_scale_reg_0.00001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --scale_reg 0.00001"
q a40 <<< "bash run.sh --raytracer_version v19_lod -s colmap/shiny_bedroom -m output_exp73/lod_init0_initscale_0.2_noscale_scale_reg_0.0000001 --init_lod_prob_0 1.0 --init_lod_scale 0.2 --lod_scale_lr 0.0 --scale_reg 0.0000001"


q a40 <<< "bash run.sh --raytracer_version v19_ -s colmap/shiny_bedroom -m output_exp73/nolod_scale_reg_0.1 --scale_reg 0.1"
q a40 <<< "bash run.sh --raytracer_version v19_ -s colmap/shiny_bedroom -m output_exp73/nolod_scale_reg_0.001 --scale_reg 0.001"
q a40 <<< "bash run.sh --raytracer_version v19_ -s colmap/shiny_bedroom -m output_exp73/nolod_scale_reg_0.00001 --scale_reg 0.00001"
q a40 <<< "bash run.sh --raytracer_version v19_ -s colmap/shiny_bedroom -m output_exp73/nolod_scale_reg_0.0000001 --scale_reg 0.0000001"