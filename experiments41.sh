

for scene in shiny_kitchen shiny_bedroom shiny_office shiny_livingroom; do 
    q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v8_baseline -s colmap/$scene -m output_exp16/$scene"
done