# scripts/train.py

import argparse
import os
import yaml
import pickle
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, hf_hub_download
from safetensors.numpy import save_file as save_safetensors
from flax.training import train_state
from transformers import AutoTokenizer
from flax.traverse_util import flatten_dict

from models.structformer_poincare import StructFormerPoincare
from utils.export_hf import export_model_to_huggingface, sanitize

load_dotenv()

def batch_data(input_ids, attention_mask, batch_size):
    for i in range(0, input_ids.shape[0], batch_size):
        yield {
            "input_ids": input_ids[i:i+batch_size],
            "attention_mask": attention_mask[i:i+batch_size],
        }

def make_milestones(max_words=100_000_000):
    return (
        list(range(1_000_000, 10_000_001, 1_000_000)) +
        list(range(20_000_000, max_words + 1, 10_000_000))
    )

def create_train_state(rng, config, vocab_size):
    model = StructFormerPoincare(
        vocab_size=vocab_size,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_length=config["seq_length"],
        c=1.0,
    )
    dummy_in = jnp.ones((1, config["seq_length"]), dtype=jnp.int32)
    dummy_mask = jnp.ones_like(dummy_in, dtype=jnp.bool_)
    params = model.init(rng, dummy_in, dummy_mask)["params"]
    tx = optax.adam(float(config["learning_rate"]))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"]).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss_fn(state.params)

@jax.jit
def eval_step(params, apply_fn, input_ids, attention_mask):
    return apply_fn({"params": params}, input_ids, attention_mask)

def eval_epoch_fast(state, input_ids, attention_mask, batch_size, n_batches=32):
    N = input_ids.shape[0]
    idx = np.random.choice(N, min(N, n_batches * batch_size), replace=False)
    ids = input_ids[idx]
    mask = attention_mask[idx]
    losses = []
    for batch in batch_data(ids, mask, batch_size):
        ids_ = jnp.array(batch["input_ids"], dtype=jnp.int32)
        msks = jnp.array(batch["attention_mask"], dtype=jnp.bool_)
        logits = eval_step(state.params, state.apply_fn, ids_, msks)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, ids_).mean()
        losses.append(float(loss))
    return np.mean(losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    HF_REPO_ID = os.environ["HF_REPO_ID"]
    create_repo(HF_REPO_ID, exist_ok=True, repo_type="model")
    api = HfApi()

    train_path = hf_hub_download(config["train_tokenized_repo"], config["train_tokenized_file"], repo_type="dataset")
    val_path = hf_hub_download(config["val_tokenized_repo"], config["val_tokenized_file"], repo_type="dataset")
    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(val_path, "rb") as f:
        val = pickle.load(f)

    train_ids, train_mask = train["input_ids"], train["attention_mask"]
    val_ids, val_mask = val["input_ids"], val["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained(config.get("vocab_name", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 42))
    state, model = create_train_state(rng, config, vocab_size)

    words_per_token = 0.75
    milestones = make_milestones()
    next_idx = 0
    words_seen = 0

    for epoch in range(config["num_epochs"]):
        print(f"\n🔁 Epoch {epoch+1}")
        epoch_loss = []

        for batch in tqdm(batch_data(train_ids, train_mask, config["batch_size"]), desc="Training"):
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, loss = train_step(state, batch_jax)
            epoch_loss.append(float(loss))

            words_seen += int(batch_jax["input_ids"].size * words_per_token)

            while next_idx < len(milestones) and words_seen >= milestones[next_idx]:
                milestone = milestones[next_idx]
                ckpt_name = f"checkpoint-{milestone // 1_000_000}M-words.safetensors"
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, ckpt_name)
                    save_safetensors(sanitize(state.params), path)
                    api.upload_file(
                        repo_id=HF_REPO_ID,
                        path_or_fileobj=path,
                        path_in_repo=f"checkpoints/{ckpt_name}",
                        commit_message=f"Checkpoint @ {milestone:,} words",
                        repo_type="model"
                    )
                next_idx += 1

        val_loss = eval_epoch_fast(state, val_ids, val_mask, config["batch_size"])
        print(f"✅ Epoch {epoch+1} → Train: {np.mean(epoch_loss):.4f} | Fast Val: {val_loss:.4f} | Words: {words_seen:,}")

    if os.getenv("HF_MODEL_EXPORT", "no").lower() == "yes":
        export_model_to_huggingface(
            state.params,
            config,
            repo_id=os.getenv("HF_REPO_ID"),
            commit_message=f"Final export after {words_seen:,} words"
        )
