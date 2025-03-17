
# for scene in shiny_livingroom; do 
#     q a40 <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene --no_densif_use_top_k -m output_eval_nodensif_v3/$scene"
# done


# for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
#     q a40 <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene --no_densif_use_top_k -m output_eval_nodensif_v3/$scene"
# done


# for scene in shiny_kitchen shiny_livingroom; do 
#     q a40 <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_v2/$scene"
# done


# for scene in shiny_bedroom shiny_office; do 
#     q a40 -name fullres_$scene <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_highres/$scene -r 1536"
# done


for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
    q a40 -n highres_1m_$scene <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_highres_1m/$scene --densif_final_num_gaussians 1000000 -r 1536"
done
