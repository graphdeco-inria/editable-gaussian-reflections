
# Try training longer & compare to a baseline also trained longer
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/gt_brdf_longtrain --iteration 90000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+gtrays_longtrain --brdf_mode static_lut --iteration 90000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --iteration 90000"

# Try "warmup": no reflection loss at first, then add it after K iterations
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --warmup 1000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --warmup 2000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --warmup 4000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --warmup 8000"
q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/lut+realrays_longtrain --brdf_mode static_lut --use_attached_brdf --warmup 16000"