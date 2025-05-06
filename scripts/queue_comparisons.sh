

for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    q a6000 -t 3:00:00 -n $scene <<< "bash run.sh -s renders/$scene -m output_comparisons/$scene -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90"
done