import argparse
import os
import sys
import yaml
import pickle
import tempfile
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi, create_repo
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax.training import train_state
from transformers import AutoTokenizer
from models.structformer_poincare import StructFormerPoincare
from nnsight.flax.inspect import trace_layers  # If installed

load_dotenv()

# Academic reference for 0.75 words/token: Brown et al. (2020, GPT-3 paper), Table 3 & A.4.

def batch_data(input_ids, attention_mask, batch_size):
    num_samples = input_ids.shape[0]
    for i in range(0, num_samples, batch_size):
        yield {
            "input_ids": input_ids[i:i+batch_size],
            "attention_mask": attention_mask[i:i+batch_size],
        }

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

def make_word_milestones(max_words=10_000_000, step=1_000_000):
    """Make a list of word-milestones per 1M up to max_words (can extend as needed)."""
    return [step * i for i in range(1, (max_words // step) + 1)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Tokenized data from HF Hub
    train_pkl_path = hf_hub_download(
        repo_id=config["train_tokenized_repo"],
        filename=config["train_tokenized_file"],
        repo_type="dataset"
    )
    val_pkl_path = hf_hub_download(
        repo_id=config["val_tokenized_repo"],
        filename=config["val_tokenized_file"],
        repo_type="dataset"
    )

    with open(train_pkl_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_pkl_path, "rb") as f:
        val_data = pickle.load(f)
    train_ids = train_data["input_ids"]        # shape: (N, seq_length)
    train_mask = train_data["attention_mask"]
    val_ids = val_data["input_ids"]
    val_mask = val_data["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained(config.get("vocab_name", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 42))
    state, model = create_train_state(rng, config, vocab_size)

    # HF Hub setup for checkpoints
    HF_REPO_ID = os.environ["HF_REPO_ID"]
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    api = HfApi()

    print(f"Training on {train_ids.shape[0]} samples, {train_ids.shape[1]} tokens per sample.")

    # BabyLM milestones and word accounting
    words_per_token = 0.75  # Brown et al., 2020 ("Language Models are Few-Shot Learners")[1]
    seq_length = train_ids.shape[1]
    batch_size = config["batch_size"]

    words_seen = 0
    milestones = make_word_milestones(max_words=10_000_000, step=1_000_000)  # Edit for longer runs
    next_milestone_idx = 0

    total_tokens = train_ids.shape[0] * train_ids.shape[1]
    est_total_words = int(total_tokens * words_per_token)
    print(f"Estimated training words in full set: {est_total_words:,}")

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        losses = []

        # Progress bar for batches
        batch_iter = tqdm(
            list(batch_data(train_ids, train_mask, batch_size)),
            total=(len(train_ids) // batch_size + 1),
            desc=f"Training (epoch {epoch+1})"
        )
        for batch_idx, batch in enumerate(batch_iter):
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, loss = train_step(state, batch_jax)
            losses.append(float(loss))

            # --- Words/Checkpointing ---
            # Each sample = seq_length tokens, so total tokens in batch:
            batch_tokens = batch_jax["input_ids"].size
            batch_words = int(batch_tokens * words_per_token)
            words_seen += batch_words

            # Upload checkpoint if passing a milestone
            while (next_milestone_idx < len(milestones)) and (words_seen >= milestones[next_milestone_idx]):
                milestone = milestones[next_milestone_idx]
                ckpt_name = f"checkpoint-{milestone//1_000_000}M-words.pkl"
                with tempfile.TemporaryDirectory() as temp_dir:
                    ckpt_path = os.path.join(temp_dir, ckpt_name)
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(state.params, f)
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=f"checkpoints/{ckpt_name}",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        commit_message=f"Checkpoint after {milestone:,} words seen"
                    )
                tqdm.write(f"☁ Uploaded {ckpt_name} at {milestone:,} words.")
                # --- NNsight trace at checkpoint ---
                try:
                    with trace_layers(StructFormerPoincare, layers=["layers.*"]) as trace:
                        dummy = jnp.ones((1, seq_length), dtype=jnp.int32)
                        mask = jnp.ones((1, seq_length), dtype=jnp.bool_)
                        _ = model.apply({"params": state.params}, dummy, mask)
                        print("🧠 NNsight trace captured at milestone checkpoint.")
                except Exception as e:
                    print(f"⚠ NNsight trace failed (optional): {e}")
                next_milestone_idx += 1

            # Print every 100 batches, as well as via tqdm
            if batch_idx % 100 == 0:
                tqdm.write(f"Batch {batch_idx}: Running train loss = {loss:.4f}, words seen = {words_seen:,}")

        avg_loss = float(np.mean(losses))
        val_loss = eval_epoch(state, val_ids, val_mask, batch_size)
        print(f"✅ Epoch {epoch+1}: Train loss = {avg_loss:.4f} | Val loss = {val_loss:.4f} | Words seen so far = {words_seen:,}")

