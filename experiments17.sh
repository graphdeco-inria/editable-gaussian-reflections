## The fit normals + position are WORSE when training is performed with the attached BRDF. Why? Is it just an issue of incomplete convergence or poor initial convergence?


# Try training longer & compare to a baseline also trained longer
# q a40 -n gt_brdf_longtrain <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/\$OAR_JOB_NAME --iteration 90000"
# q a40 -n lut+gtrays_longtrain <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/\$OAR_JOB_NAME --brdf_mode static_lut --iteration 90000"
# q a40 -n lut+realrays_longtrain <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_longtrain/\$OAR_JOB_NAME --brdf_mode static_lut --use_attached_brdf --iteration 90000"

# Try "warmup": no reflection loss at first, then add it after K iterations

# for i in 1000; do  # 2000 4000 8000 16000
#     q a40 -n warmup_${i} <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_warmup/\$OAR_JOB_NAME --brdf_mode static_lut --use_attached_brdf --warmup ${i}"
# done

# Try different ray offsets. It could be that the reflection rays hit the guassians from inside the plate
for offset in 0.0 0.1 0.2 0.4 0.8; do
    q a40 -n offset_${offset} <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset ${offset} --brdf_mode static_lut --use_attached_brdf"
done