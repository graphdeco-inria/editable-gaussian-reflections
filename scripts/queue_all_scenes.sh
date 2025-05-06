

# for scene in multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2 multichromeball_kitchen_v2; do 
#     q a40 -t 1:00:00 -n $scene <<< "bash run.sh -s renders/$scene -m output_v90_with_pruning/$scene -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90"
# done

for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    q a40 -t 1:00:00 -n $scene <<< "bash run.sh -s renders/$scene -m output_v90_sanity_check_ablations/$scene -r 256 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90"
done