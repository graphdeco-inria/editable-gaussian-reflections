ITERATIONS=10000
OUTDIR=output_current_state_v0.4b
LABEL=v0.4b_10k

# ----------------

mkdir -p dump
for path in /home/ypoirier/gaussian-splatting-lambertian/$OUTDIR/*; do     
    NAME=$(basename $path);     
    cp $path/train_view/iter_$(printf "%09d" $ITERATIONS)_4.png dump/${NAME}_${LABEL}.png;     
    cp $path/train_view/iter_$(printf "%09d" $ITERATIONS)_4_normal.png dump/${NAME}_normal_${LABEL}.png;     
    cp $path/train_view/iter_$(printf "%09d" $ITERATIONS)_4_position.png dump/${NAME}_position_${LABEL}.png;     
    cp $path/train_view/iter_$(printf "%09d" $ITERATIONS)_4_F0.png dump/${NAME}_F0_${LABEL}.png; 
done