import argparse
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

from models.structformer_poincare import StructformerPoincare
from configs.config import StructformerConfig
from utils.save_utils import (
    batch_data,
    make_milestones,
    save_checkpoint_branch,
    export_model_to_huggingface,
)
from utils.logging_utils import (
    init_wandb,
    init_tensorboard,
    log_metrics_wandb,
    log_metrics_tensorboard,
    plot_losses,
)


def create_train_state(rng, config_dict, vocab_size):
    model = StructformerPoincare(
        vocab_size=vocab_size,
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_length=config_dict["seq_length"],
        c=config_dict.get("c", 1.0),
    )
    dummy = jnp.ones((1, config_dict["seq_length"]), dtype=jnp.int32)
    mask = jnp.ones_like(dummy, dtype=jnp.bool_)
    params = model.init(rng, dummy, mask)["params"]
    tx = optax.adam(config_dict["learning_rate"])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"])
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"]).mean()

        # Placeholder for Poincaré loss - you should replace with actual computation from your model
        poincare_loss = 0.0
        total_loss = ce_loss + 0.01 * poincare_loss

        return total_loss, (ce_loss, poincare_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, poincare_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, float(ce_loss), float(poincare_loss)


@jax.jit
def eval_step(params, apply_fn, input_ids, attention_mask):
    logits = apply_fn({"params": params}, input_ids, attention_mask)
    return optax.softmax_cross_entropy_with_integer_labels(logits, input_ids).mean()


def eval_epoch(state, ds, batch_size, max_batches=32):
    # Sample subset for validation
    num_samples = min(len(ds), batch_size * max_batches)
    idx = np.random.choice(len(ds), size=num_samples, replace=False)
    losses = []
    for start in range(0, num_samples, batch_size):
        batch_idx = idx[start : start + batch_size]
        batch = ds.select(batch_idx)
        x = jnp.array(batch["input_ids"], dtype=jnp.int32)
        m = jnp.array(batch["attention_mask"], dtype=jnp.bool_)
        loss = eval_step(state.params, state.apply_fn, x, m)
        losses.append(float(loss))
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    HF_REPO_ID = os.environ["HF_REPO_ID"]
    HF_TOKENIZED_DATASET_REPO = os.environ["HF_TOKENIZED_DATASET_REPO"]

    # Initialize logging
    wandb_run = None
    tb_writer = None
    if args.use_wandb:
        wandb_run = init_wandb(project_name=config.get("wandb_project", "structformer-flax"), run_name=config.get("wandb_run_name"), config=config)
    if args.use_tensorboard:
        tb_writer = init_tensorboard()

    # Load dataset splits
    ds = load_dataset(HF_TOKENIZED_DATASET_REPO)
    train_ds = ds["train"]
    if "dev" in ds:
        val_ds = ds["dev"]
    elif "validation" in ds:
        val_ds = ds["validation"]
    else:
        val_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(config["vocab_name"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 0))
    state, model = create_train_state(rng, config, vocab_size)

    milestones = make_milestones()
    words_seen = 0
    next_idx = 0
    global_step = 0

    train_losses, val_losses, poincare_losses, val_steps = [], [], [], []

    print(f"✨ Starting training with {len(train_ds):,} training and {len(val_ds):,} validation examples")

    for epoch in range(config["num_epochs"]):
        print(f"\n🔁 Epoch {epoch + 1}")

        # Train loop
        for i in tqdm(range(0, len(train_ds), config["batch_size"]), desc="Training batches"):
            batch = train_ds.select(range(i, min(i + config["batch_size"], len(train_ds))))
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, ce_loss, p_loss = train_step(state, batch_jax)

            train_losses.append(ce_loss)
            poincare_losses.append(p_loss)
            global_step += 1

            # Logging to wandb and tensorboard if enabled
            metrics = {"train/ce_loss": ce_loss, "train/poincare_loss": p_loss}
            if wandb_run:
                log_metrics_wandb(metrics, global_step)
            if tb_writer:
                log_metrics_tensorboard(tb_writer, metrics, global_step)

            # Count words seen (approximate), assuming 0.75 tokens per word as in GPT-2
            nonpad_count = (np.array(batch["input_ids"]) != pad_id).sum()
            words_seen += int(nonpad_count * 0.75)

            # Checkpoint milestones
            while next_idx < len(milestones) and words_seen >= milestones[next_idx]:
                branch_name = f"checkpoint-{milestones[next_idx] // 1_000_000}M-words"
                print(f"\n📤 Pushing checkpoint branch: {branch_name}")
                save_checkpoint_branch(
                    params=state.params,
                    config=config,
                    branch_name=branch_name,
                    repo_id=HF_REPO_ID,
                    include_modeling_files=["modeling.py", "structformer_config.py", "hyperbolic_layers.py"],
                )
                next_idx += 1

        # Validation at epoch end
        val_loss = eval_epoch(state, val_ds, config["batch_size"])
        val_losses.append(val_loss)
        val_steps.append(global_step)

        print(f"✅ Epoch {epoch+1} complete | Train CE Loss: {ce_loss:.4f} | Validation Loss: {val_loss:.4f} | Words Seen: {words_seen:,}")

        # Log validation loss
        if wandb_run:
            log_metrics_wandb({"validation/loss": val_loss}, global_step)
        if tb_writer:
            log_metrics_tensorboard(tb_writer, {"validation/loss": val_loss}, global_step)

    # Save final model to main branch (optional)
    if os.getenv("HF_MODEL_EXPORT", "no").lower() == "yes":
        export_model_to_huggingface(
            params=state.params,
            config=config,
            repo_id=HF_REPO_ID,
            commit_message=f"Final model after {words_seen:,} words",
            include_modeling_files=["modeling.py", "structformer_config.py", "hyperbolic_layers.py"],
        )

    # Plot and save losses as PDF
    plot_losses(
        train_losses=train_losses,
        val_losses=val_losses,
        poincare_losses=poincare_losses,
        val_steps=val_steps,
        save_path="loss_graph.pdf",
    )

    # Close logging handlers
    if wandb_run:
        wandb_run.finish()
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
