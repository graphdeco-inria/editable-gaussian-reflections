



for w in 2.0; do 
    q a40 <<< "F0_LOSS_WEIGHT=$w GLOSSY_LOSS_WEIGHT=0.0 python3 train.py --raytracer_version 1_bounce_lut_t0.001 -s colmap/hard_kitchen_mirror -m output_exp4/f0_weight_$w"
done