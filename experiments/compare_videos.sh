#!/bin/bash

for object in bunny cube sphere; do 
    for material in shiny smooth metal mirror; do 
        ffmpeg -y \
            -i output/${object}_${material}_baseline_no_sh/00/test/ours_30000/renders/%05d.png \
            -i output/${object}_${material}_baseline/00/test/ours_30000/renders/%05d.png \
            -i output/${object}_${material}/00/test/ours_30000/renders/%05d.png \
            -i output/${object}_${material}_baseline_no_sh/00/test/ours_30000/gt/%05d.png \
            -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4" -c:v libx264 -pix_fmt yuv420p \
            compare_${object}_${material}.mp4
    done
done

