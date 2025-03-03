

bash run.sh --raytracer_version v17_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp65/bedroom_lod_init_scale_0.2_nodiff --lod_init_scale 0.2 --lod_mean_lr 0.000 --lod_scale_lr 0.000
bash run.sh --raytracer_version v17_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp65/bedroom_lod_init_scale_0.2_nodiff --lod_init_scale 0.4 --lod_mean_lr 0.000 --lod_scale_lr 0.000
bash run.sh --raytracer_version v17_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp65/bedroom_lod_init_scale_0.2_nodiff --lod_init_scale 0.6 --lod_mean_lr 0.000 --lod_scale_lr 0.000
bash run.sh --raytracer_version v17_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp65/bedroom_lod_init_scale_0.2_meandiff --lod_init_scale 0.2 --lod_mean_lr 0.005 --lod_scale_lr 0.000
bash run.sh --raytracer_version v17_lod -s colmap/shiny_bedroom --glossy_loss_weight 0.001 -m output_exp65/bedroom_lod_init_scale_0.2_scalediff --lod_init_scale 0.2 --lod_mean_lr 0.005 --lod_scale_lr 0.005



        