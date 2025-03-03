

# is the "popping" in and out caused by too few gaussians? it doesn't seem fixable by switching to local sort...

for frac in 0.2 0.4 0.6 0.8 1.0; do 
    q a6000 <<< "bash run.sh --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp97/init_frac_$frac --prob_blur_targets 1.0 --densify_from_iter 999999 --lod_init_frac_extra_points $frac"
done


# Can we make densification work with the existing mechanism?

for weight in 0.25 0.5 1.0 2.0; do 
    q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp9b7/densify_lod_weight_$weight --prob_blur_targets 1.0 --densif_lod_ranking_weight $weight"
done
