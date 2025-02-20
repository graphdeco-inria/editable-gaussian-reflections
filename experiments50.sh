



for scene in bedroom kitchen livingroom office; do q a40 <<< "bash run.sh --raytracer_version v15_0bounce_noblur -s colmap/shiny_${scene} --glossy_loss_weight 0.001 -m output_exp50/shiny_${scene} --densif_jitter_clones"; done 


for minsize in 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.0010; for scene in bedroom; do q a40 <<< "bash run.sh --raytracer_version v15_0bounce_noblur -s colmap/shiny_$scene --glossy_loss_weight 0.001 -m output_exp50/shiny_${scene}_minsize_${minsize} --min_gaussian_size $minsize"; done; done 




