
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
#     q a40 -n fullres_$scene <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_highres/$scene -r 1536"
# done


# for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
#     q a40 -n highres_4m_$scene <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_highres_4m/$scene --densif_final_num_gaussians 4000000"
# done

# for scene in shiny_bedroom shiny_kitchen shiny_livingroom shiny_office; do 
#     q a40 -n highres_4m_$scene <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_eval_highres_8m/$scene --densif_final_num_gaussians 8000000"
# done


for scene in shiny_livingroom; do 
    for m in 1 2 4 8 16; do
        q a40 -n fullres_${m}m_${scene} <<< "bash run.sh --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_num_gaussians/${m}m_$scene -r 1536 --densif_final_num_gaussians ${m}000000"
    done
done