

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------





q a40 <<< "bash run.sh --raytracer_version v19 -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_baseline"
q a40 <<< "bash run.sh --raytracer_version v19_unweighted_rgb -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_unweighted_rgb"
q a40 <<< "bash run.sh --raytracer_version v19_unweighted_rgb_and_opacity -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_unweighted_rgb_and_opacity"

q a40 <<< "bash run.sh --raytracer_version v19_2bounce -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_baseline_2bounce"
q a40 <<< "bash run.sh --raytracer_version v19_unweighted_rg_2bounce -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_unweighted_rgb_2bounce"
q a40 <<< "bash run.sh --raytracer_version v19_unweighted_rgb_and_opacity_2bounce -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp72/shiny_bedroom_unweighted_rgb_and_opacity_2bounce"






q a40 <<< "bash run.sh --raytracer_version v19 -s colmap/shiny_bedroom --glossy_loss_weight 0.01 -m output_exp72/shiny_bedroom_baseline_0.01"
q a40 <<< "bash run.sh --raytracer_version v19 -s colmap/shiny_bedroom --glossy_loss_weight 0.1 -m output_exp72/shiny_bedroom_baseline_0.1"
