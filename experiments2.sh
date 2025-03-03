
q a40 <<< "python train.py -s renders/chromeball_kitchen_vanilla --split_spec_diff --num_farfield_init_points 1000 --label brdf_1k_points"
q a40 <<< "python train.py -s renders/chromeball_kitchen_vanilla --split_spec_diff --num_farfield_init_points 2000 --label brdf_2k_points"
q a40 <<< "python train.py -s renders/chromeball_kitchen_vanilla --split_spec_diff --num_farfield_init_points 4000 --label brdf_4k_points"
q a40 <<< "python train.py -s renders/chromeball_kitchen_vanilla --split_spec_diff --num_farfield_init_points 8000 --label brdf_8k_points"
q a40 <<< "python train.py -s renders/chromeball_kitchen_vanilla --split_spec_diff --num_farfield_init_points 16000 --label brdf_16k_points"