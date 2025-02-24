

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


ZEROLOSSES="--diffuse_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --specular_loss_weight 0.0 --albedo_loss_weight 0.0 --metalness_loss_weight 0.0"

q a40 <<< "bash run.sh --raytracer_version v18_globalsort -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp80/globalsort_baseline $ZEROLOSSES"
q a40 <<< "bash run.sh --raytracer_version v18_globalsort -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp80/globalsort_10xmeanlr --position_lr_init 0.0016 --position_lr_final 0.000016 $ZEROLOSSES"
q a40 <<< "bash run.sh --raytracer_version v18_globalsort -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp80/globalsort_100xmeanlr --position_lr_init 0.016 --position_lr_final 0.00016 $ZEROLOSSES"
q a40 <<< "bash run.sh --raytracer_version v18_globalsort -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/globalsort_1000xmeanlr --position_lr_init 0.16 --position_lr_final 0.0016 $ZEROLOSSES"


