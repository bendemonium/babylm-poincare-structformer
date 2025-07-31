set -e

python3.11 -m venv bb-env
source bb-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

huggingface-cli login