
for MODE in static_lut finetuned_lut disabled gt; do
    q a40 -n ${MODE}_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_brdf/${MODE}_attached --brdf_mode $MODE --use_attached_brdf";
done

for MODE in disabled gt static_lut finetuned_lut; do
    q a40 -n $MODE <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf  -s colmap/hard_kitchen_mirror/ -m output_brdf/$MODE --brdf_mode $MODE";
done


