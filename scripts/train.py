import argparse
import os
import yaml
import pickle
import tempfile
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi, create_repo
from safetensors.flax import save_file as save_safetensors
from flax.traverse_util import flatten_dict, unflatten_dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax.training import train_state
from transformers import AutoTokenizer

from models.structformer_poincare import StructFormerPoincare

load_dotenv()

# ------------------------- Utilities -------------------------

def batch_data(input_ids, attention_mask, batch_size):
    for i in range(0, input_ids.shape[0], batch_size):
        yield {
            "input_ids": input_ids[i:i+batch_size],
            "attention_mask": attention_mask[i:i+batch_size],
        }

def make_milestones(max_words=10_000_000, step=1_000_000):
    return [step * i for i in range(1, (max_words // step) + 1)]

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file as save_safetensors  # ← use numpy backend!
from flax.traverse_util import flatten_dict

def sanitize_and_save(params, path):
    flat = flatten_dict(jax.tree_util.tree_map(lambda x: x, params), sep="/")
    safe_flat = {}

    for k, v in flat.items():
        if isinstance(v, (np.ndarray, jnp.ndarray)) and v.dtype != object:
            key = k if isinstance(k, str) else "/".join(k)
            safe_flat[key] = np.array(v)  # Cast to np.ndarray for safety
        else:
            print(f"⚠️ Skipping {k} due to invalid dtype: {getattr(v, 'dtype', type(v))}")

    save_safetensors(safe_flat, path)



# ------------------------- Model Logic -------------------------

def create_train_state(rng, config, vocab_size):
    model = StructFormerPoincare(
        vocab_size=vocab_size,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_length=config["seq_length"],
        c=1.0,
    )
    dummy_input = jnp.ones((1, config["seq_length"]), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, config["seq_length"]), dtype=jnp.bool_)
    params = model.init(rng, dummy_input, dummy_mask)["params"]
    tx = optax.adam(float(config["learning_rate"]))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"])
        labels = batch["input_ids"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    loss = loss_fn(state.params)
    return state, loss

def eval_epoch(state, input_ids, attention_mask, batch_size):
    losses = []
    for batch in tqdm(batch_data(input_ids, attention_mask, batch_size), desc="Validating", leave=False):
        batch = {
            "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
            "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
        }
        logits = state.apply_fn({"params": state.params}, batch["input_ids"], batch["attention_mask"])
        labels = batch["input_ids"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        losses.append(float(loss))
    return np.mean(losses)

# ------------------------- Main -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    HF_REPO_ID = os.environ["HF_REPO_ID"]
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    api = HfApi()

    # Load tokenized data
    train_path = hf_hub_download(
        repo_id=config["train_tokenized_repo"],
        filename=config["train_tokenized_file"],
        repo_type="dataset"
    )
    val_path = hf_hub_download(
        repo_id=config["val_tokenized_repo"],
        filename=config["val_tokenized_file"],
        repo_type="dataset"
    )

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)

    train_ids = train_data["input_ids"]
    train_mask = train_data["attention_mask"]
    val_ids = val_data["input_ids"]
    val_mask = val_data["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained(config.get("vocab_name", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 42))
    state, model = create_train_state(rng, config, vocab_size)

    print(f"🧃 Training on {len(train_ids)} examples ({len(train_ids) * config['seq_length']:,} tokens)")
    print(f"📏 Val set size: {len(val_ids)} examples ({len(val_ids) * config['seq_length']:,} tokens)")

    # BabyLM tracking: words seen
    words_per_token = 0.75
    words_seen = 0
    milestones = make_milestones(max_words=10_000_000, step=1_000_000)
    next_milestone_idx = 0

    for epoch in range(config["num_epochs"]):
        print(f"\n🔁 Epoch {epoch+1}/{config['num_epochs']}")
        losses = []

        for batch_idx, batch in enumerate(tqdm(batch_data(train_ids, train_mask, config["batch_size"]), desc="Training")):
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, loss = train_step(state, batch_jax)
            losses.append(float(loss))

            # Track words seen
            batch_tokens = batch_jax["input_ids"].size
            words_seen += int(batch_tokens * words_per_token)

            # Check milestone
            while next_milestone_idx < len(milestones) and words_seen >= milestones[next_milestone_idx]:
                milestone = milestones[next_milestone_idx]
                ckpt_base = f"checkpoint-{milestone // 1_000_000}M-words"
                with tempfile.TemporaryDirectory() as tmpdir:
                    st_path = os.path.join(tmpdir, ckpt_base + ".safetensors")
                    sanitize_and_save(state.params, st_path)
                    api.upload_file(
                        path_or_fileobj=st_path,
                        path_in_repo=f"checkpoints/{ckpt_base}.safetensors",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        commit_message=f"Checkpoint after {milestone:,} words (safetensors)"
                    )
                tqdm.write(f"☁️ Uploaded {ckpt_base}.safetensors after {milestone:,} words.")
                next_milestone_idx += 1

        # End of epoch - evaluate
        train_loss = float(np.mean(losses))
        val_loss = eval_epoch(state, val_ids, val_mask, config["batch_size"])
        print(f"✅ Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Words seen = {words_seen:,}")

