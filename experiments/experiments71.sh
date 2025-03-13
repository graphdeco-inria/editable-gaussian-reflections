

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp71/shiny_bedroom"
q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_kitchen --glossy_loss_weight 0.001 -m output_exp71/shiny_kitchen"
q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_livingroom --glossy_loss_weight 0.001 -m output_exp71/shiny_livingroom"
q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_office --glossy_loss_weight 0.001 -m output_exp71/shiny_office"


