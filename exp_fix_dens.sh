# python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v32_tonemap/ -s colmap/shiny_kitchen -m output_fix_dots/baseline -r 384


# python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v32_tonemap/ -s colmap/shiny_kitchen -m output_fix_dots/tinyres -r 192


# python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v32_tonemap/ -s colmap/shiny_kitchen -m output_fix_dots/disable_until_5k -r 384 --disable_glossy_until_iter 5000


# python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v32_tonemap/ -s colmap/hard_kitchen_mirror -m output_fix_dots/baseline_mirror -r 384




python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.1 -s colmap/hard_kitchen_mirror -m output_fix_dots_2/minnorm0.1 -r 384

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.3 -s colmap/hard_kitchen_mirror -m output_fix_dots_2/minnorm0.3 -r 384

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.6 -s colmap/hard_kitchen_mirror -m output_fix_dots_2/minnorm0.6 -r 384

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.3 -s colmap/hard_kitchen_mirror -m output_fix_dots_2/minnorm0.3_until5k -r 384 --disable_glossy_until_iter 5000




python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.1 -s colmap/hard_kitchen_mirror -m output_fix_dots_3/disable_until_5k -r 384 --disable_glossy_until_iter 5000

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.1 -s colmap/hard_kitchen_mirror -m output_fix_dots_3/disable_until_5k_w0.0001 -r 384 --disable_glossy_until_iter 5000 --glossy_loss_weight 0.0001

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.1 -s colmap/hard_kitchen_mirror -m output_fix_dots_3/disable_until_5k_w0.00001 -r 384 --disable_glossy_until_iter 5000 --glossy_loss_weight 0.00001

python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v33_normalminnorm0.1 -s colmap/hard_kitchen_mirror -m output_fix_dots_3/disable_until_10k -r 384 --disable_glossy_until_iter 10000

