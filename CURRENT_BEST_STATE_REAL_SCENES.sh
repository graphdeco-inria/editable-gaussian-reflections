PRIOR_DATASET_TRAIN_ONLY=1 DIFFUSE_IS_RENDER=1 SKIP_EXTRA_INIT=1 python train.py -s priors/refnerf_priors/toycar -m output_debug/toycar_diff_is_render_from_transform_json6 --raytracer_version /home/ypoirier/optix/gausstracer/build/v69 --no_znear_init_pruning --no_znear_densif_pruning

--> THERE'S AN ASPECT RATIO DIFFERENCE REMAINING 





# for chrome balls:

# FORCE_EXPOSURE=1 SKIP_TONEMAPPING_INPUT=1 python train.py -s renders/multichromeball_identical_kitchen_v2 -m output2/mcv2_identical_notonemap3 --raytracer_version /home/ypoirier/optix/gausstracer/build/v69_notonemap --densify_from 500 -r 512