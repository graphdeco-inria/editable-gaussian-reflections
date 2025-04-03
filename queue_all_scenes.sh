for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    q a6000 <<< "bash run.sh -s colmap/$scene -m output_v45/$scene -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v45"
done