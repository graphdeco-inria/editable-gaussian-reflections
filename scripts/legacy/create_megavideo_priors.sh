
DIR_PRIORS=${1?You must pass in a parent directory containing training runs with priors (first arg) and regular training runs (second arg)}
DIR_REGULAR=${2?You must pass in a parent directory containing training runs without priors (first arg) and regular training runs (second arg)}

files=($DIR_PRIORS/shiny*/videos_regular/test_renders_hq.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp1.mp4

files=($DIR_REGULAR/shiny*/test_comparison_hq_regular.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp2.mp4

ffmpeg -y -i tmp1.mp4 -i tmp2.mp4 -filter_complex "hstack=inputs=2" megavideo_priors_$DIR.mp4

rm tmp1.mp4
rm tmp2.mp4

files=($DIR_PRIORS/shiny*/videos_env_rot_1/test_diffuse_renders_hq.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp1.mp4

files=($DIR_REGULAR/shiny*/videos_env_rot_1/test_diffuse_renders_hq.mp4)
ffmpeg -y $(printf -- "-i %s " "${files[@]}") -filter_complex "vstack=inputs=${#files[@]}" tmp2.mp4

ffmpeg -y -i tmp1.mp4 -i tmp2.mp4 -filter_complex "hstack=inputs=2" megavideo_priors_envrot_$DIR.mp4

rm tmp1.mp4
rm tmp2.mp4