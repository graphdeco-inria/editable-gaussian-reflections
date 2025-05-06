set -e 

DIR=${1?You must pass in a parent directory containing multiple training runs e.g. output/}

for scene in shiny_kitchen; do 

  # Add no ablations
  choices=(NONE)
  videos=($DIR/${scene}_ablate_NONE/videos_regular/test_renders_hq.mp4 "${videos[@]}")
  labels=("NO PREDICTIONS")

  # Add first row
  first_row_ablations=(
      DISENTANGLEMENT
      GEOMETRY
      F0
      ROUGHNESS
  )
  choices+=("${first_row_ablations[@]}")
  for choice in "${first_row_ablations[@]}"; do
      videos+=("$DIR/${scene}_ablate_${choice}/videos_regular/test_renders_hq.mp4")
  done
  for choice in "${first_row_ablations[@]}"; do
    labels+=("PREDICTED ${choice}")
  done

  # Add ground truth
  choices+=(GROUND_TRUTH)
  videos+=($DIR/${scene}_ablate_NONE/videos_regular/test_gts_hq.mp4)
  labels+=(GROUND_TRUTH)

  # Add second row
  second_row_ablations=(
      DISENTANGLEMENT+GEOMETRY
      DISENTANGLEMENT+ROUGHNESS+F0
      GEOMETRY+ROUGHNESS+F0
      DISENTANGLEMENT+GEOMETRY+ROUGHNESS+F0
  )
  choices+=("${second_row_ablations[@]}")
  for choice in "${second_row_ablations[@]}"; do
      videos+=("$DIR/${scene}_ablate_${choice}/videos_regular/test_renders_hq.mp4")
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

  ffmpeg -y -i tmp_fullrow.mp4 \
  -filter_complex "[0:v]crop=${half_width}:${height}:0:0[top];[0:v]crop=${half_width}:${height}:${half_width}:0[bottom];[top][bottom]vstack=inputs=2" \
  -c:v libx264 -crf 17 -preset slow ablations_$scene.mp4

  rm tmp_fullrow.mp4

done