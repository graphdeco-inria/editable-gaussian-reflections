

PRUNING_SIGMA=0 python train.py -s renders/shiny_office_with_book -m output_teaser/try28 --raytracer_version /home/ypoirier/optix/raytracer/build/v81_zplanes -r 256  --loss_weight_glossy 1.0  --val_views 66 135