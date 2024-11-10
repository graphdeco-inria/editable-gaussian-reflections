

# q a40 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen_mirror/ -m output/mirror_weight_1.0 --exposure 5"
# q a40 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen_one_bounce/ -m output/one_bounce_weight_1.0 --exposure 10"

# q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py -s colmap/hard_kitchen_mirror/ -m output/mirror_weight_0.1 --exposure 5"
# q a40 <<< "REFLECTION_LOSS_WEIGHT=0.1 python train.py -s colmap/hard_kitchen_one_bounce/ -m output/one_bounce_weight_0.1 --exposure 10"

# q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py -s colmap/hard_kitchen_mirror/ -m output/mirror_weight_0.0 --exposure 5"
# q a40 <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py -s colmap/hard_kitchen_one_bounce/ -m output/one_bounce_weight_0.0 --exposure 10"

# q a40 <<< "REFLECTION_LOSS_WEIGHT=4.0 python train.py -s colmap/hard_kitchen_mirror/ -m output/mirror_weight_4.0 --exposure 5"
# q a40 <<< "REFLECTION_LOSS_WEIGHT=4.0 python train.py -s colmap/hard_kitchen_one_bounce/ -m output/one_bounce_weight_4.0 --exposure 10"


q a40 -n mirror_one_bounce_weight_1.0 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen_mirror_one_bounce/ -m output/mirror_one_bounce_weight_1.0 --exposure 10"
q a40 -n mirror_one_bounce_weight_0.1 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen_mirror_one_bounce/ -m output/mirror_one_bounce_weight_0.1 --exposure 10"
q a40 -n mirror_one_bounce_weight_0.0 <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py -s colmap/hard_kitchen_mirror_one_bounce/ -m output/mirror_one_bounce_weight_0.0 --exposure 10"
q a40 -n output/mirror_one_bounce_weight_4.0 <<< "REFLECTION_LOSS_WEIGHT=4.0 python train.py -s colmap/hard_kitchen_mirror_one_bounce/ -m output/mirror_one_bounce_weight_4.0 --exposure 10"


q a40 -n regular_weight_1.0 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen/ -m output/regular_weight_1.0 --exposure 5"
q a40 -n regular_weight_0.1 <<< "REFLECTION_LOSS_WEIGHT=1.0 python train.py -s colmap/hard_kitchen/ -m output/regular_weight_0.1 --exposure 5"
q a40 -n output/regular_weight_0.0 <<< "REFLECTION_LOSS_WEIGHT=0.0 python train.py -s colmap/hard_kitchen/ -m output/regular_weight_0.0 --exposure 5"
q a40 -n output/regular_weight_4.0 <<< "REFLECTION_LOSS_WEIGHT=4.0 python train.py -s colmap/hard_kitchen/ -m output/regular_weight_4.0 --exposure 5"



