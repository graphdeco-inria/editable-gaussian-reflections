



q a40 <<< "bash run.sh -s colmap/shiny_kitchen -m output_current_state/finalize -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v45"

q a40 <<< "CHECK_NAN=1 python train.py -s colmap/shiny_kitchen -m output_current_state/finalize_checknan -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v45"

q a40 <<< "CHECK_NAN=1 python train.py -s colmap/shiny_kitchen -m output_current_state/finalize_checknan_split --no_densif_no_splitting -r 768 --raytracer_version /home/ypoirier/optix/gausstracer/build/v45"