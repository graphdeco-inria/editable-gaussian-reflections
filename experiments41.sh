

# shiny_office shiny_kitchen
for scene in shiny_bedroom shiny_livingroom; do 
    q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 render.py --raytracer_version v8_baseline -s colmap/$scene -m output_exp16/$scene"
done

GLOSSY_LOSS_WEIGHT=0.001 python3 render.py --raytracer_version v8_baseline -s colmap/shiny_livingroom -m output_exp16/shiny_livingroom






q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v8_baseline -s colmap/multichromeball_identical_kitchen -m output_exp16/baseline_multichromeball_identical"

q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v8_baseline -s colmap/multichromeball_kitchen -m output_exp16/baseline_multichromeball"