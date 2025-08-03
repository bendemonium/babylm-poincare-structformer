import argparse
import os
import yaml
import pickle
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, hf_hub_download
from safetensors.numpy import save_file as save_safetensors
from flax.training import train_state
from flax.traverse_util import flatten_dict
from transformers import AutoTokenizer

from models.structformer_poincare import StructFormerPoincare
from utils.export_hf import sanitize, export_model_to_huggingface

load_dotenv()


def batch_data(ids, masks, batch_size):
    for i in range(0, ids.shape[0], batch_size):
        yield {
            "input_ids": ids[i:i+batch_size],
            "attention_mask": masks[i:i+batch_size],
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
    dummy = jnp.ones((1, config["seq_length"]), dtype=jnp.int32)
    mask = jnp.ones_like(dummy, dtype=jnp.bool_)
    params = model.init(rng, dummy, mask)["params"]
    tx = optax.adam(config["learning_rate"])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"]).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss_fn(state.params)

def eval_step(params, apply_fn, input_ids, attention_mask):
    # Run the model forward given frozen weights
    return apply_fn({"params": params}, input_ids, attention_mask)
eval_step_jit = jax.jit(eval_step, static_argnames=["apply_fn"])

def eval_epoch_fast(state, input_ids, attention_mask, batch_size, n_batches=32):
    total_examples = input_ids.shape[0]
    sample_size = min(n_batches * batch_size, total_examples)
    selected_indices = np.random.choice(total_examples, size=sample_size, replace=False)

    sample_input_ids = input_ids[selected_indices]
    sample_attention_mask = attention_mask[selected_indices]

    losses = []
    for batch in tqdm(
        batch_data(sample_input_ids, sample_attention_mask, batch_size),
        desc="Validating (subset)", leave=False
    ):
        x = jnp.array(batch["input_ids"], dtype=jnp.int32)
        m = jnp.array(batch["attention_mask"], dtype=jnp.bool_)
        logits = eval_step_jit(state.params, state.apply_fn, x, m)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, x).mean()
        losses.append(float(loss))

    return np.mean(losses)


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
    train_path = hf_hub_download(config["train_tokenized_repo"], config["train_tokenized_file"], repo_type="dataset")
    val_path = hf_hub_download(config["val_tokenized_repo"], config["val_tokenized_file"], repo_type="dataset")
    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(val_path, "rb") as f:
        val = pickle.load(f)

    train_ids, train_mask = train["input_ids"], train["attention_mask"]
    val_ids, val_mask = val["input_ids"], val["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained(config["vocab_name"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 42))
    state, model = create_train_state(rng, config, vocab_size)

    print(f"🧃 Training on {len(train_ids)} examples ({len(train_ids) * config['seq_length']:,} tokens)")
    print(f"📏 Val set: {len(val_ids)} examples")

    milestones = make_milestones()
    next_idx = 0
    words_seen = 0

    for epoch in range(config["num_epochs"]):
        print(f"\n🔁 Epoch {epoch+1}")
        losses = []

        for batch in tqdm(batch_data(train_ids, train_mask, config["batch_size"]), desc="Training"):
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, loss = train_step(state, batch_jax)
            losses.append(float(loss))

            # Count only non-pad tokens for words_seen
            non_padded = (np.array(batch_jax["input_ids"]) != pad_id).sum()
            words_in_batch = int(non_padded * 0.75)  # GPT2 estimate
            words_seen += words_in_batch

            while next_idx < len(milestones) and words_seen >= milestones[next_idx]:
                milestone = milestones[next_idx]
                ckpt_name = f"checkpoint-{milestone // 1_000_000}M-words.safetensors"
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, ckpt_name)
                    save_file = sanitize(state.params)
                    save_safetensors(save_file, path)
                    api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=f"checkpoints/{ckpt_name}",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        commit_message=f"Checkpoint @ {milestone:,} words"
                    )
                tqdm.write(f"☁️ Saved {ckpt_name} at {milestone:,} words")
                next_idx += 1

        val_loss = eval_epoch_fast(state, val_ids, val_mask, config["batch_size"])
        print(f"✅ Epoch {epoch+1}: Train Loss = {np.mean(losses):.4f} | Val Loss ≈ {val_loss:.4f} | Words Seen = {words_seen:,}")

    if os.getenv("HF_MODEL_EXPORT", "no").lower() == "yes":
        export_model_to_huggingface(
            state.params,
            config,
            repo_id=HF_REPO_ID,
            commit_message=f"Final model after {words_seen:,} words"
        )

# python scripts/train.py --config configs/base.yaml