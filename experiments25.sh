



q a40 -n aa_baseline <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_aa_pooling/baseline --brdf_mode static_lut"

q a40 -n aa_randpool <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_aa_pooling/baseline --brdf_mode static_lut --random_pool_props"




