
q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11_baseline -s colmap/shiny_kitchen -m output_exp17/baseline_v11_kitchen"
q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11 -s colmap/shiny_kitchen -m output_exp17/v11_kitchen"

q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11_baseline -s colmap/shiny_bedroom -m output_exp17/baseline_v11_bedroom"
q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11 -s colmap/shiny_bedroom -m output_exp17/v11_bedroom"

q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11_baseline -s colmap/shiny_office -m output_exp17/baseline_v11_office"
q a40 <<< "GLOSSY_LOSS_WEIGHT=0.001 python3 train.py --raytracer_version v11 -s colmap/shiny_office -m output_exp17/v11_office"
