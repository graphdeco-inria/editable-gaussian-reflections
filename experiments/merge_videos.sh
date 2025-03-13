
for run in output/*_diffuse; do 
    name=$(basename $run)
    # ffmpeg -y -i $(echo output/$name/*/test/ours_30000/renders/)%05d.png \
    #        -i $(echo output/${name/_diffuse//}/*/test/ours_30000/renders/)%05d.png \
    #        -i $(echo output/$name/*/test/ours_30000/gt/)%05d.png \
    #        -filter_complex "[0:v][1:v][2:v]hstack=inputs=3" \
    #        -c:v libx264 -pix_fmt yuv420p \
    #        output/$name/${name}_combined.mp4
    convert output/$name/*/iter_000030000_train_view_4.jpg -crop 50%x100%+0+0 output/$name/cropped.png
    convert output/$name/cropped.png output/${name/_diffuse//}/*/iter_000030000_train_view_4.jpg +append output/$name/${name}_train.png
done

