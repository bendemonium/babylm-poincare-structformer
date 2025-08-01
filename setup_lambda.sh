
# nano ~/.ssh/config


set -e

sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv bb-env
source bb-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# git clone https://github.com/bendemonium/babylm-poincare-structformer.git
cd babylm-poincare-structformer

huggingface-cli login

git config --global user.name "bendemonium"
git config --global user.email "ridhib2422@gmail.com"
git config --global credential.helper store

git remote -v

git add .

