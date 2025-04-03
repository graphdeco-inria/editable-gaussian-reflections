python train.py -s colmap/shiny_kitchen -m output_sample_brdf/sample_brdf -r 512  --raytracer_version /home/ypoirier/optix/gausstracer/build/v38_sample_brdf

python train.py -s colmap/shiny_kitchen -m output_sample_brdf/no_sample_brdf -r 512 --raytracer_version /home/ypoirier/optix/gausstracer/build/v38_no_sample_brdf

python train.py -s colmap/shiny_kitchen -m output_sample_brdf/no_sample_brdf_small_init -r 512 --init_scale_factor 0.2 --raytracer_version /home/ypoirier/optix/gausstracer/build/v38_no_sample_brdf

python train.py -s colmap/shiny_kitchen -m output_sample_brdf/sample_no_brdf_2bounce -r 512 --raytracer_version /home/ypoirier/optix/gausstracer/build/v38_no_sample_brdf_2bounce
