
echo "Downloading all datasets to data/ ..."
mkdir -p data

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/renders.zip -O data/renders.zip
python -m zipfile -e data/renders.zip data/ && rm data/renders.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/renders_priors.zip -O data/renders_priors.zip
python -m zipfile -e data/renders_priors.zip data/ && rm data/renders_priors.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/neural_catacaustics.zip -O data/neural_catacaustics.zip
python -m zipfile -e data/neural_catacaustics.zip data/ && rm data/neural_catacaustics.zip &

wget https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/datasets/demos.zip -O data/demos.zip
python -m zipfile -e data/demos.zip data/ && rm data/demos.zip &

wait

echo "All datasets downloaded and extracted to data/."