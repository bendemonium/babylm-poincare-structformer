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
from flax import struct
import argparse

from models.structformer_poincare import StructformerPoincare
from models.hyperbolic_layers import poincare_distance
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
def train_step(state, batch, c=1.0):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"])
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"]).mean()

        # Poincaré loss: average pairwise distance between token embeddings and their neighbors
        emb = logits[..., :-1, :]  # [B, T-1, D]
        tgt = logits[..., 1:, :]   # [B, T-1, D]

        dist = poincare_distance(emb, tgt, c=c)
        poincare_loss = dist.mean()

        total_loss = ce_loss + 0.01 * poincare_loss
        return total_loss, (ce_loss, poincare_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, poincare_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, ce_loss, poincare_loss


@jax.jit
def eval_step(params, apply_fn, input_ids, attention_mask):
    logits = apply_fn({"params": params}, input_ids, attention_mask)
    return optax.softmax_cross_entropy_with_integer_labels(logits, input_ids).mean()


def eval_epoch(state, val_data, batch_size, dataset_type="hf", max_batches=32):
    losses = []

    if dataset_type == "pickle":
        input_ids = val_data["input_ids"]
        attention_mask = val_data["attention_mask"]
        total_samples = len(input_ids)
        num_samples = min(total_samples, batch_size * max_batches)
        indices = np.random.choice(total_samples, num_samples, replace=False)

        for start in range(0, num_samples, batch_size):
            idx = indices[start : start + batch_size]
            x = jnp.array(input_ids[idx], dtype=jnp.int32)
            m = jnp.array(attention_mask[idx], dtype=jnp.bool_)
            loss = eval_step(state.params, state.apply_fn, x, m)
            losses.append(float(loss))

    else:
        total_samples = len(val_data)
        num_samples = min(total_samples, batch_size * max_batches)
        indices = np.random.choice(total_samples, num_samples, replace=False)

        for start in range(0, num_samples, batch_size):
            idx = indices[start : start + batch_size]
            batch = val_data.select(idx)
            x = jnp.array(batch["input_ids"], dtype=jnp.int32)
            m = jnp.array(batch["attention_mask"], dtype=jnp.bool_)
            loss = eval_step(state.params, state.apply_fn, x, m)
            losses.append(float(loss))

    return np.mean(losses) if losses else float("inf")


def load_pickle_from_hub(repo_id, filename):
    from huggingface_hub import hf_hub_download
    import pickle

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, choices=["hf", "pickle"], default="hf")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    HF_REPO_ID = os.environ["HF_REPO_ID"]
    HF_TOKENIZED_DATASET_REPO = os.environ.get("HF_TOKENIZED_DATASET_REPO")

    # Logging setup
    wandb_run = init_wandb(config.get("wandb_project", "structformer"), config.get("wandb_run_name"), config) if args.use_wandb else None
    tb_writer = init_tensorboard() if args.use_tensorboard else None

    if args.dataset_type == "hf":
        ds = load_dataset(HF_TOKENIZED_DATASET_REPO)
        train_ds = ds["train"]
        val_ds = ds.get("dev") or ds.get("validation") or ds["test"]
    else:
        assert config.get("train_tokenized_repo") and config.get("train_tokenized_file"), "Missing train pickle config"
        assert config.get("val_tokenized_repo") and config.get("val_tokenized_file"), "Missing val pickle config"
        train_ds = load_pickle_from_hub(config["train_tokenized_repo"], config["train_tokenized_file"])
        val_ds = load_pickle_from_hub(config["val_tokenized_repo"], config["val_tokenized_file"])

    # Print dataset sizes
    if args.dataset_type == "pickle":
        print(f"✅ Loaded training dataset with {len(train_ds['input_ids']):,} examples")
        print(f"✅ Loaded validation dataset with {len(val_ds['input_ids']):,} examples")
    else:
        print(f"✅ Loaded training dataset with {len(train_ds):,} examples")
        print(f"✅ Loaded validation dataset with {len(val_ds):,} examples")

    tokenizer = AutoTokenizer.from_pretrained(config["vocab_name"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    rng = jax.random.PRNGKey(config.get("seed", 0))
    state, _ = create_train_state(rng, config, vocab_size)

    milestones = make_milestones()
    words_seen = 0
    global_step = 0
    next_idx = 0

    train_losses, val_losses, poincare_losses, val_steps = [], [], [], []

    for epoch in range(config["num_epochs"]):
        print(f"\n🔁 Epoch {epoch + 1}")
        if args.dataset_type == "pickle":
            train_ids, train_mask = train_ds["input_ids"], train_ds["attention_mask"]
        else:
            train_ids = train_ds["input_ids"]
            train_mask = train_ds["attention_mask"]

        for batch in tqdm(batch_data(train_ids, train_mask, config["batch_size"]), desc="Training batches"):
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.bool_),
            }
            state, ce_loss, p_loss = train_step(state, batch_jax, c=config.get("c", 1.0))
            ce_loss = float(ce_loss)
            p_loss = float(p_loss)

            train_losses.append(ce_loss)
            poincare_losses.append(p_loss)
            global_step += 1

            if wandb_run:
                log_metrics_wandb({"train/ce_loss": ce_loss, "train/poincare_loss": p_loss}, global_step)
            if tb_writer:
                log_metrics_tensorboard(tb_writer, {"train/ce_loss": ce_loss, "train/poincare_loss": p_loss}, global_step)

            nonpad = (np.array(batch["input_ids"]) != pad_id).sum()
            words_seen += int(nonpad * 0.75)

            while next_idx < len(milestones) and words_seen >= milestones[next_idx]:
                branch = f"checkpoint-{milestones[next_idx] // 1_000_000}M-words"
                print(f"\n📤 Pushing checkpoint branch: {branch}")
                save_checkpoint_branch(
                    params=state.params,
                    config=config,
                    branch_name=branch,
                    repo_id=HF_REPO_ID,
                    include_modeling_files=["hyperbolic_layers.py", "structformer_config.py"],
                )
                next_idx += 1

        val_loss = eval_epoch(state, val_ds, config["batch_size"], args.dataset_type)
        val_losses.append(val_loss)
        val_steps.append(global_step)

        print(f"✅ Epoch {epoch + 1} complete | Train CE Loss: {ce_loss:.4f} | Val Loss: {val_loss:.4f} | Words Seen: {words_seen:,}")

        if wandb_run:
            log_metrics_wandb({"validation/loss": val_loss}, global_step)
        if tb_writer:
            log_metrics_tensorboard(tb_writer, {"validation/loss": val_loss}, global_step)

    if os.getenv("HF_MODEL_EXPORT", "no").lower() == "yes":
        export_model_to_huggingface(
            params=state.params,
            config=config,
            repo_id=HF_REPO_ID,
            commit_message=f"Final model after {words_seen:,} words",
            include_modeling_files=["hyperbolic_layers.py", "structformer_config.py"],
        )

    plot_losses(
        train_losses=train_losses,
        val_losses=val_losses,
        poincare_losses=poincare_losses,
        val_steps=val_steps,
        save_path="loss_graph.pdf",
    )

    if wandb_run:
        wandb_run.finish()
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
