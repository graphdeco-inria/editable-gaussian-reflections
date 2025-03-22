
NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/multichromeball_identical_kitchen -m output_chromeball/v1 -r 768 

NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30 -s colmap/multichromeball_identical_kitchen -m output_chromeball/v1_noglossloss -r 768 --glossy_loss_weight 0.0

NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays -s colmap/multichromeball_identical_kitchen -m output_chromeball/v2_nolosses -r 384 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0

NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays_t0.0001 -s colmap/multichromeball_identical_kitchen -m output_chromeball/v2_nolosses_t0.0001 -r 384 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0


NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays -s colmap/multichromeball_identical_kitchen -m output_chromeball/v2_nolosses_nodensif -r 384 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --no_densif_use_top_k


SKIP_CLAMP_MINSIZE=1 NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays -s colmap/multichromeball_identical_kitchen -m output_chromeball/v2_nolosses_nodensif_skipclamp -r 384 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --no_densif_use_top_k


SKIP_CLAMP_MINSIZE=1 NO_TONEMAPPING=1 python3 train.py --raytracer_version /home/ypoirier/optix/gausstracer/build/v30_1bounce_gtrays_t0.0001_noest/ -s colmap/multichromeball_identical_kitchen -m output_chromeball/v2_nolosses_nodensif_skipclamp_noest -r 384 --glossy_loss_weight 0.0 --normal_loss_weight 0.0 --position_loss_weight 0.0 --f0_loss_weight 0.0 --roughness_loss_weight 0.0 --no_densif_use_top_k




