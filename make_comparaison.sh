
ffmpeg -y -i output/sphere_metal/01/test_sliced/ours_30000/renders/%05d.png -i output/sphere_metal_dynamic/01/test_sliced/ours_30000/renders/%05d.png -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p mixed.mp4
