set -e 

DIR=${1?You must pass in a parent directory containing multiple training runs e.g. output/}


for render_pass in _ _diffuse_ _glossy_; do 
  for scene in shiny_{bedroom,kitchen,livingroom,bedroom}; do 

    # Add ground truth
    choices=(GROUND_TRUTH)
    videos=($DIR/${scene}_ablate_NONE/videos_regular/test${render_pass}gts_hq.mp4)
    labels=(GROUND_TRUTH)

    # Add no ablations
    choices+=(NONE)
    videos+=($DIR/${scene}_ablate_NONE/videos_regular/test${render_pass}renders_hq.mp4)
    labels+=("NO PREDICTIONS")

    # Add first row
    first_row_ablations=(
        DISENTANGLEMENT
        GEOMETRY
        BRDF
    )
    choices+=("${first_row_ablations[@]}")
    for choice in "${first_row_ablations[@]}"; do
        videos+=("$DIR/${scene}_ablate_${choice}/videos_regular/test${render_pass}renders_hq.mp4")
    done
    for choice in "${first_row_ablations[@]}"; do
      labels+=("PREDICTED ${choice}")
    done

    # Add ground truth again
    choices+=(GROUND_TRUTH)
    videos+=($DIR/${scene}_ablate_NONE/videos_regular/test${render_pass}gts_hq.mp4)
    labels+=(GROUND_TRUTH)

    # Add second row
    second_row_ablations=(
        DISENTANGLEMENT+GEOMETRY
        DISENTANGLEMENT+BRDF
        GEOMETRY+BRDF
        DISENTANGLEMENT+GEOMETRY+BRDF
    )
    choices+=("${second_row_ablations[@]}")
    for choice in "${second_row_ablations[@]}"; do
        videos+=("$DIR/${scene}_ablate_${choice}/videos_regular/test${render_pass}renders_hq.mp4")
    done
    for choice in "${second_row_ablations[@]}"; do
      labels+=("PREDICTED ${choice}")
    done

    fontfile=/usr/share/fonts/dejavu/DejaVuSans.ttf 
    ffmpeg -y $(for i in "${videos[@]}"; do echo -n "-i $i "; done) \
    -crf 17 -preset slow -filter_complex "$(
      for i in "${!choices[@]}"; do
        echo -n "[${i}:v]drawtext=text='${labels[$i]}':x=10:y=10:fontsize=11:fontcolor=white:fontfile=${fontfile}:borderw=2:bordercolor=black[${i}l]; "
      done
      for i in "${!choices[@]}"; do
        echo -n "[${i}l]"
      done
      echo "hstack=inputs=${#choices[@]}[out]"
    )" -map "[out]" tmp_fullrow.mp4

    width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "${videos[0]}")
    height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "${videos[0]}")
    row_width=$((10 * width))
    half_width=$((row_width / 2))

    filename=ablations_${scene}_${render_pass//_/}.mp4
    filename=${filename/_.mp4/.mp4}
    ffmpeg -y -i tmp_fullrow.mp4 \
    -filter_complex "[0:v]crop=${half_width}:${height}:0:0[top];[0:v]crop=${half_width}:${height}:${half_width}:0[bottom];[top][bottom]vstack=inputs=2" \
    -c:v libx264 -crf 17 -preset slow $filename

    rm tmp_fullrow.mp4

  done
done