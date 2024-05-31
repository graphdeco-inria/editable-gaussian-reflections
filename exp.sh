
for scene in datasets/objects/*; do
    q a40 -- LABEL=_baseline_25_views bash run.sh $scene --keep_every_kth_view 8
    q a40 -- LABEL=_ours_25_views bash run_split.sh $scene --keep_every_kth_view 8
    q a40 -- LABEL=_baseline_no_sh_25_views bash run.sh $scene --diffuse_only --keep_every_kth_view 8
done

# 

for scene in datasets/objects/*; do
    q a40 -- LABEL=_baseline_50_views bash run.sh $scene --keep_every_kth_view 4
    q a40 -- LABEL=_ours_50_views bash run_split.sh $scene --keep_every_kth_view 4
    q a40 -- LABEL=_baseline_no_sh_50_views bash run.sh $scene --diffuse_only --keep_every_kth_view 4
done

# #

# todo power of 2 sequence, add a --max
