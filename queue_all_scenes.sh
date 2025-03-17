for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
    q a40 <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene --no_densif_use_top_k -m output_eval_nodensif/$scene"
done

for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
    q a40 <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval/$scene"
done
