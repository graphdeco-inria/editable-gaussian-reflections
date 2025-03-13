



for weight in 10 100 1000 10000 100000 1000000; do
    q a40 <<< "python train.py --raytracer_version v23_t0.001 -s colmap/shiny_bedroom -m output_exp101/weight_$weight --prob_blur_targets 1.0 --use_diffuse_target --densif_lod_ranking_weight $weight"
done