
# Try "warmup": no reflection loss at first, then add it after K iterations
for i in 1000 2000 4000 8000 16000; do 
    q a40 -n warmup_${i} <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_warmup/\$OAR_JOB_NAME --brdf_mode static_lut --use_attached_brdf --warmup ${i}"
done

# Try different ray offsets. It could be that the reflection rays hit the guassians from inside the plate!
for offset in 0.0 0.1 0.2 0.4 0.8; do
    q a40 -n offset_${offset} <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_offset/\$OAR_JOB_NAME --ray_offset ${offset} --brdf_mode static_lut --use_attached_brdf"
done