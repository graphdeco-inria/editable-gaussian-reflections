
echo "Downloading all datasets to data/ ..."
mkdir -p data

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/renders.zip -O data/renders.zip
python -m zipfile -e data/renders.zip data/renders &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/renders_priors.zip -O data/renders_priors.zip
python -m zipfile -e data/renders_priors.zip data/renders_priors &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/neural_catacaustics.zip -O data/neural_catacaustics.zip
python -m zipfile -e data/neural_catacaustics.zip data/neural_catacaustics &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/demos.zip -O data/demos.zip
python -m zipfile -e data/demos.zip data/demos &

wait

rm data/*.zip
echo "All datasets downloaded and extracted to data/."