### just some stuff to
help me set up the mood, set up the franchise hahaha

on lambda :/

start off with the python configs first

```
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

then the cloning comes in

```
git clone https://github.com/bendemonium/babylm-poincare-structformer.git
```

!!! cd 2 the cloned directory


then set up env

```
python3.10 -m venv bb-env
source bb-env/bin/activate
source bb-env/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"
pip uninstall numpy -y
pip install "numpy<2.0"
pip install -r requirements.txt
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

git stuff

```
git config --global user.name "bendemonium"
git config --global user.email "ridhib2422@gmail.com"
git config --global credential.helper store

git remote -v

git add .

sudo apt install git-lfs
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
```

hf login 

```
huggingface-cli login
```

!!! copy .env file onto the root, the export

```
export $(grep -v '^#' .env | xargs)
```

testin the utils
```
python -m utils.test_utils
```

training dry runs

```
# cheap dry run
python scripts/train.py \
  --config configs/dry_structformer_only.yaml \
  --dry_run_structformer_only \
  --output_dir runs/dry_structformer_only
```

```
# full dry run
python scripts/train.py \
  --config configs/dry_full_poincare.yaml \
  --dry_run_structformer_poincare \
  --output_dir runs/dry_structformer_poincare
```

full training in this bitch
```
python scripts/train.py \
  --config configs/base.yaml \
  --output_dir runs/full_structformer_only
```