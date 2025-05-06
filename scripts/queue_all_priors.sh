


for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    q a6000 -t 1:00:00 -n ${scene}_priors <<< "bash run.sh -s priors/real_datasets_v1/renders_priors/$scene -m output_priors_fullres/$scene -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90 ${@:3}"
done

