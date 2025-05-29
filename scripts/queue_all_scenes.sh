

for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom multichromeball_identical_kitchen_v2 multichromeball_tint_kitchen_v2 multichromeball_kitchen_v2 shiny_office_with_book; do 
    q a6000 -t 2:00:00 -n final_$scene <<< "bash run.sh -s renders/${scene} -m output_v100_final_v17/${scene} -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v100-submit-nohack"
done  