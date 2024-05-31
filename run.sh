: ${1?specify a dataset path}

set -e

label=$(basename $1)$LABEL
version=${VERSION:-$(printf "%02d\n" $(ls -1 output/$label 2>/dev/null | wc -l))}

outfile=output/$label/$version
mkdir -p tmp

{
    # Draw text labels

    convert -size 800x200 xc:black  -pointsize 65 -fill white -gravity center -draw "text 0,0 'PREDICTION'" tmp/label_pred.png
    convert -size 800x200 xc:black  -pointsize 65 -fill white -gravity center -draw "text 0,0 'GROUND TRUTH'" tmp/label_gt.png
    convert tmp/label_pred.png tmp/label_gt.png +append tmp/label_test.png

    convert -size 800x200 xc:black  -pointsize 65 -fill white -gravity center -draw "text 0,0 'PREDICTION SLICE'" tmp/label_slice.png
    convert -size 800x200 xc:black  -pointsize 65 -fill white -gravity center -draw "text 0,0 'SLICE + FIXED CAM'" tmp/label_slice_fixed.png
    convert tmp/label_slice.png tmp/label_slice_fixed.png +append tmp/label_sliced.png
}

[ -z "$SKIP_TRAIN" ] && python train.py --label $label/$version -s "${@}"

[ -z "$SKIP_RENDER" ] && python render.py --skip_train -m $outfile --eval
ffmpeg -y -i $outfile/test/ours_30000/renders/%05d.png -i $outfile/test/ours_30000/gt/%05d.png -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p $outfile/unlabled.mp4
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $outfile/unlabled.mp4)
ffmpeg -i tmp/label_test.png -i $outfile/unlabled.mp4 -y -loop 1 -t 3 -pix_fmt yuv420p -filter_complex "[0:v][1:v]vstack=inputs=2" $outfile/test_all.mp4
rm $outfile/unlabled.mp4

[ -z "$SKIP_RENDER" ] && python render.py --skip_train -m $outfile --eval --sliced
ffmpeg -y -i $outfile/test_sliced/ours_30000/renders/%05d.png -c:v libx264 -pix_fmt yuv420p $outfile/test_sliced.mp4

[ -z "$SKIP_RENDER" ] && python render.py --skip_train -m $outfile --eval --sliced --fixed_pov
ffmpeg -y -i $outfile/test_sliced_fixed/ours_30000/renders/%05d.png -c:v libx264 -pix_fmt yuv420p $outfile/test_sliced_fixed.mp4

ffmpeg -y -i $outfile/test_sliced.mp4 -i $outfile/test_sliced_fixed.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p $outfile/unlabled.mp4
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $outfile/unlabled.mp4)
ffmpeg -i tmp/label_sliced.png -i $outfile/unlabled.mp4 -y -loop 1 -t 3 -pix_fmt yuv420p -filter_complex "[0:v][1:v]vstack=inputs=2" $outfile/test_sliced_all.mp4
rm $outfile/unlabled.mp4


