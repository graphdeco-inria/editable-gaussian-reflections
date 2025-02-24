

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_baseline"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_opacity_decay_0.9999 --opacity_decay 0.9999"
q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_opacity_decay_0.999 --opacity_decay 0.999"
q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_opacity_decay_0.99 --opacity_decay 0.99"

q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_scale_decay_0.999 --scale_decay 0.999"
q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_scale_decay_0.9999 --scale_decay 0.9999"
q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_scale_decay_0.99999 --scale_decay 0.99999"
q a40 <<< "bash run.sh --raytracer_version v20_lod_0bounce -s colmap/shiny_bedroom -m output_exp79/nolod_0bounce_scale_decay_0.999999 --scale_decay 0.999999"

# ----------------------------------------
