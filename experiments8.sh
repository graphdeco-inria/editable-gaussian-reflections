

# try increasing the fixed finetuning
q a6000 -n static_lut_detached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_static_v2/detached --brdf_mode static_lut";

q a6000 -n static_lut_attached <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py --raytracer_version build_v0.1_attached_brdf -s colmap/hard_kitchen_mirror/ -m output_fix_static_v2/attached --brdf_mode static_lut --use_attached_brdf";


# try increasing the position loss
for LOSS in 4.0 16.0; do
    q a6000 -n pos_loss_$LOSS <<< "REFLECTION_LOSS_WEIGHT=0.1 POSITION_LOSS_WEIGHT=$LOSS python train.py --raytracer_version build_v0.1_attached_brdf  -s colmap/hard_kitchen_mirror/ -m output_fix_position/pos_loss_${LOSS} --brdf_mode static_lut --use_attached_brdf";
done

