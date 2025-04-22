for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    q a40 <<< "bash run.sh -s renders/$scene -m output_v74/$scene -r 512 --raytracer_version /home/ypoirier/optix/gausstracer/build/v74"
done

for scene in multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_value_kitchen_v2; do 
    q a40 <<< "bash run.sh -s renders/$scene -m output_v74/$scene -r 512 --raytracer_version /home/ypoirier/optix/gausstracer/build/v74"
done