
SRC=${1?Error: No source path given} # e.g. output/multichromeball_fixsh/test/ours_30000

ffmpeg -y -pattern_type glob -i $SRC'/glossy_renders/*.png' -c:v libx264 -pix_fmt yuv420p -crf 18 $SRC/glossy_pred_video.mp4 &&
ffmpeg -y -pattern_type glob -i $SRC'/glossy_gt/*.png' -c:v libx264 -pix_fmt yuv420p -crf 18 $SRC/glossy_gt_video.mp4 &&
ffmpeg -y  -i $SRC/glossy_pred_video.mp4 -i $SRC/glossy_gt_video.mp4 -crf 18  -filter_complex hstack glossy_video.mp4