set -e

bash run_all_synthetic.sh "$@"
bash run_all_synthetic_priors.sh "$@"
bash run_all_demos.sh "$@"
bash run_all_neural_catacaustics.sh "$@"