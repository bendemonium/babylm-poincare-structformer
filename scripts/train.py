import argparse, yaml, os
import utils.env
import numpy as np
import pickle
import jax, optax
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from transformers import AutoTokenizer
from models.structformer_poincare import StructFormerPoincare

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
    dummy_input_ids = jnp.ones((1, config["seq_length"]), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((1, config["seq_length"]), dtype=jnp.bool_)
    params = model.init(rng, dummy_input_ids, dummy_attention_mask)["params"]
    tx = optax.adam(config["learning_rate"])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Paths
    train_pkl_file = config.get("train_tokenized_file", "data/train_tokenized.pkl")
    vocab_name = config.get("vocab_name", "gpt2")

    # Create output/checkpoint dir using config file name
    run_name = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(vocab_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load tokenized data
    with open(train_pkl_file, "rb") as f:
        data = pickle.load(f)
    input_ids = data["input_ids"]      # shape: (num_examples, seq_length)
    attention_mask = data["attention_mask"]

    rng = jax.random.PRNGKey(config.get("seed", 42))
    state, model = create_train_state(rng, config, vocab_size)

    print(f"Training on {input_ids.shape[0]} samples, {input_ids.shape[1]} tokens per sample.")

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        batch_losses = []
        for batch_idx, batch in enumerate(batch_data(input_ids, attention_mask, config["batch_size"])):
            # Convert numpy to jax arrays
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, loss = train_step(state, batch_jax)
            batch_losses.append(loss)
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {float(loss):.4f}")
        avg_loss = float(jnp.mean(jnp.stack(batch_losses)))
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        checkpoints.save_checkpoint(ckpt_dir, state.params, step=epoch+1, overwrite=True)
    print(f"Training finished. Checkpoints saved in '{ckpt_dir}'")
