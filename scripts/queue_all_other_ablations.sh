
for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom multichromeball_identical_kitchen_v2 multichromeball_kitchen_v2; do 
    # Ablate the disentanglement
    q a40 -t 1:00:00 <<< "bash run.sh -s renders/$scene -m output_other_ablations/disentanglement/$scene -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 --regular_loss_weight 5.0 --diffuse_loss_weight 0.0 --glossy_loss_weight 0.0 --rebalance_losses_at_iter -1 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1"

    # Ablates the schedule
    q a40 -t 1:00:00 <<< "bash run.sh -s renders/$scene -m output_other_ablations/schedule/$scene -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 --rebalance_losses_at_iter -1 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1"

    # Abalate the loss weighting
    q a40 -t 1:00:00 <<< "bash run.sh -s renders/$scene -m output_other_ablations/loss_weight/$scene -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 --glossy_loss_weight 5.0 --diffuse_loss_weight 5.0"

    # Ablate the schedule + the loss weighting
    q a40 -t 1:00:00 <<< "bash run.sh -s renders/shiny_kitchen -m output_other_ablations/schedule_and_loss_weight/shiny_kitchen -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 --rebalance_losses_at_iter -1 --no_bounces_until_iter -1 --max_one_bounce_until_iter -1 --glossy_loss_weight 5.0 --diffuse_loss_weight 5.0"
done
