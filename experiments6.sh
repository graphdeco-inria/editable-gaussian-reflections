
# for OFFSET in 0.0 0.001 0.01 0.1 0.5 1.0; do q a40 -n offset_$OFFSET <<< "REFLECTION_LOSS_WEIGHT=4.0 python train.py -s colmap/hard_kitchen_mirror/ -m output/offset_$OFFSET --ray_offset $OFFSET"; done



# for OFFSET in 0.0 0.001 0.01 0.1 0.5 1.0; do q a40 -n offset_$OFFSET <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py -s colmap/hard_kitchen_mirror/ -m output_ray_offset_weight_0.1/offset_$OFFSET --ray_offset $OFFSET"; done