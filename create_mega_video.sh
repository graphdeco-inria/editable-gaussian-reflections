
DIR=${1?You must pass in a parent directory containing multiple training runs e.g. output/}

files=($DIR/multichrome*/test_comparison_hq_regular.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp1.mp4

files=($DIR/shiny*/test_comparison_hq_regular.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp2.mp4

ffmpeg -y -i tmp1.mp4 -i tmp2.mp4 -filter_complex "hstack=inputs=2" megavideo_$DIR.mp4

rm tmp1.mp4
rm tmp2.mp4

files=($DIR/multichrome*/videos_env_rot_1/test_diffuse_renders_hq.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp1.mp4

files=($DIR/shiny*/videos_env_rot_1/test_diffuse_renders_hq.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp2.mp4

ffmpeg -y -i tmp1.mp4 -i tmp2.mp4 -filter_complex "hstack=inputs=2" megavideo_envrot_$DIR.mp4

rm tmp1.mp4
rm tmp2.mp4