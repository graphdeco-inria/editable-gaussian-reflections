set -e

python train.py -m "$@"
python render_novel_views.py -m "$@"

ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/diffuse/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/diffuse.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/specular/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/glossy.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/render/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/render.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/normal/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/normal.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/depth/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/depth.mp4"