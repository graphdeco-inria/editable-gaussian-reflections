


q a6000 -n ours_gtrays <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays_eps1e-3 -s colmap/$scene -m output_too_dark/gtrays_eps1e-3 -r 768"

v31_tonemap



q a6000 -n fix_tonemap <<< "python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v31_tonemap -s colmap/$scene -m output_too_dark/fixtonemap -r 384"
