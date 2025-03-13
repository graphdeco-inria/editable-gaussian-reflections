
q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_nolod_baseline"
q a40 <<< "bash run.sh --raytracer_version v18_globalsort_a0.001 -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_globalsort_a0.001"
q a40 <<< "bash run.sh --raytracer_version v18_globalsort_a0.001_tthresh0 -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_globalsort_a0.001_tthresh0"

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


ZEROLOSSES="--diffuse_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --specular_loss_weight 0.0 --albedo_loss_weight 0.0 --metalness_loss_weight 0.0"

q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_nolod_baseline_onlygloss $ZEROLOSSES"

q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/onlygloss_10x --position_lr_init 0.0016 --position_lr_final 0.000016 $ZEROLOSSES"

q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_nolod_baseline_100xmeanlr --position_lr_init 0.016 --position_lr_final 0.00016 $ZEROLOSSES"

q a40 <<< "bash run.sh --raytracer_version v18_nolod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp68/v18_nolod_baseline_1000xmeanlr --position_lr_init 0.16 --position_lr_final 0.0016 $ZEROLOSSES"


