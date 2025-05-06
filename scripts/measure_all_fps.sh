
for scene in shiny_kitchen shiny_office shiny_livingroom shiny_bedroom; do 
    python measure_fps.py -s renders/$scene -m output_comparisons/$scene -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v90
done