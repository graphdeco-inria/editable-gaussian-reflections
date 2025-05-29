# set -e

REAL_SCENE=1 python train.py --position_loss_weight 0.0 -m "${@}" 
ZNEAR=1.0 REAL_SCENE=1 python render_novel_views.py -m "${@}" 

ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/diffuse/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/DIFFUSE.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/glossy/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/GLOSSY.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/render/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/RENDER.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/normal/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/NORMAL.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/depth/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/DEPTH.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/ray_origin/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/RAY_ORIGIN.mp4"
ffmpeg -y -framerate 30 -pattern_type glob -i "$1/novel_views/ours_8000/ray_direction/*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$1/novel_views/RAY_DIRECTION.mp4"