# Ablate the disentanglement
bash run.sh -s renders/multichromeball_tint_kitchen_v2 -m output_v91_disentanglement_ablation/multichromeball_tint_kitchen_v2 -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v91 --regular_loss_weight 5.0 --diffuse_loss_weight 0.0 --glossy_loss_weight 0.0 --rebalance_losses_at_iter 999999999999 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1

# Ablates the schedule
bash run.sh -s renders/multichromeball_tint_kitchen_v2 -m output_v91_schedule_ablation/multichromeball_tint_kitchen_v2 -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v91 --rebalance_losses_at_iter -1 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1

# Abalate the loss weighting
bash run.sh -s renders/multichromeball_tint_kitchen_v2 -m output_v91_loss_weight_ablation/multichromeball_tint_kitchen_v2 -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v91 --glossy_loss_weight 5.0 --diffuse_loss_weight 5.0

# Ablate the schedule + the loss weighting
bash run.sh -s renders/shiny_kitchen -m output_v91_loss_and_schedule_ablation/shiny_kitchen -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v91 --rebalance_losses_at_iter -1 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1 --glossy_loss_weight 5.0 --diffuse_loss_weight 5.0


