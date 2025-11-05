
echo "Downloading all models to pretrained/ ..."
mkdir -p pretrained

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_kitchen_no_eval.zip -O pretrained/shiny_kitchen_no_eval.zip
python -m zipfile -e pretrained/shiny_kitchen_no_eval.zip pretrained/ && rm pretrained/shiny_kitchen_no_eval.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_office_no_eval.zip -O pretrained/shiny_office_no_eval.zip
python -m zipfile -e pretrained/shiny_office_no_eval.zip pretrained/ && rm pretrained/shiny_office_no_eval.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_livingroom_no_eval.zip -O pretrained/shiny_livingroom_no_eval.zip
python -m zipfile -e pretrained/shiny_livingroom_no_eval.zip pretrained/ && rm pretrained/shiny_livingroom_no_eval.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/multibounce_pre-edited.zip -O pretrained/multibounce_pre-edited.zip
python -m zipfile -e pretrained/multibounce_pre-edited.zip pretrained/ && rm pretrained/multibounce_pre-edited.zip &

wait

echo "All models downloaded and extracted to pretrained/."