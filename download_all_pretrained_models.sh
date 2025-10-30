
echo "Downloading all datasets to pretrained/ ..."
mkdir -p pretrained

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_kitchen.zip -O pretrained/shiny_kitchen.zip
python -m zipfile -e pretrained/shiny_kitchen.zip pretrained/shiny_kitchen &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_kitchen.zip -O pretrained/shiny_livingroom.zip
python -m zipfile -e pretrained/shiny_office.zip pretrained/shiny_office &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/shiny_office.zip -O pretrained/shiny_livingroom.zip
python -m zipfile -e pretrained/shiny_livingroom.zip pretrained/shiny_livingroom &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/multibounce_pre-edited.zip -O pretrained/multibounce_pre-edited.zip
python -m zipfile -e pretrained/multibounce_pre-edited.zip pretrained/multibounce_pre-edited &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/pretrained/multibounce_pre-edited.zip -O pretrained/multibounce_raw.zip
python -m zipfile -e pretrained/multibounce_raw.zip pretrained/multibounce_raw &

wait

rm pretrained/*.zip
echo "All datasets downloaded and extracted to pretrained/."