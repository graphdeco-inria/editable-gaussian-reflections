
scene=hard_kitchen_mirror

q a6000 <<< "NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/$scene -m output_fixtonemap/1bounce -r 768"

q a6000 <<< "NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_lut -s colmap/$scene -m output_fixtonemap/1bounce_lut -r 768"

q a6000 <<< "NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_2bounce -s colmap/$scene -m output_fixtonemap/2bounce_lut -r 768"