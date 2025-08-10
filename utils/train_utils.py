from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from flax.core import freeze, unfreeze
import optax

from models.hyperbolic_layers import poincare_distance, project_to_ball


# -------------------------------------------------
# State creation: separate embedding vs other params
# -------------------------------------------------
def create_train_states(rng, model, vocab_size, seq_len, lr_ce, lr_riem):
    """
    Initialize params and return two TrainStates:
    - state_embed: only embedding parameters
    - state_other: all other parameters
    """
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, seq_len), dtype=jnp.bool_)

    init_params = model.init(rng, dummy_input, dummy_mask)["params"]

    params_unfrozen = unfreeze(init_params)
    # Keep embeddings separate
    embed_params = {"token_embed": params_unfrozen["token_embed"]}
    other_params = {k: v for k, v in params_unfrozen.items() if k != "token_embed"}

    state_embed = train_state.TrainState.create(
        apply_fn=model.apply,
        params=freeze(embed_params),
        tx=optax.sgd(lr_riem)  # simple SGD; we project embeddings manually after update
    )

    state_other = train_state.TrainState.create(
        apply_fn=model.apply,
        params=freeze(other_params),
        tx=optax.adam(lr_ce)
    )

    return state_embed, state_other


# -------------------------------------------------
# Loss functions
# -------------------------------------------------
def compute_ce_loss(model, params_embed, params_other, batch):
    """Cross-entropy loss for MLM or next-token prediction."""
    # Merge params for a full forward
    full_params = freeze({**unfreeze(params_embed), **unfreeze(params_other)})
    logits = model.apply({"params": full_params},
                         batch["input_ids"],
                         batch.get("attention_mask", None))
    labels = batch["input_ids"]
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels
    ).mean()
    return ce_loss


def compute_poincare_loss(model, params_embed, params_other, batch, c):
    """
    Compute Poincaré distance loss using complete parameter set.
    """
    # Combine both parameter sets for model.apply()
    combined_params = {**params_embed, **params_other}
    
    # Get logits from full forward pass
    logits = model.apply({"params": combined_params},
                        batch["input_ids"], 
                        batch["attention_mask"])
    
    # Extract embeddings from the model's embedding layer
    # We'll compute distances on the logits (which represent learned representations)
    emb_current = logits[:, :-1, :]  # [B, T-1, D]
    emb_next = logits[:, 1:, :]      # [B, T-1, D] 
    
    from models.hyperbolic_layers import poincare_distance
    distances = poincare_distance(emb_current, emb_next, c=c)
    
    return distances.mean()

# -------------------------------------------------
# Train step
# -------------------------------------------------
@partial(jax.jit, static_argnames=('model',))
def train_step(state_embed, state_other, batch, c, lambda_poincare, model):
    """
    One training step:
    - CE loss: updates non-embedding params
    - Poincaré loss: updates embedding params
    - Embeddings projected back into ball
    """
    
    # Cross-entropy grads for non-embedding params
    def ce_loss_fn(params_other):
        return compute_ce_loss(model, state_embed.params, params_other, batch)

    grads_other = jax.grad(ce_loss_fn)(state_other.params)
    state_other = state_other.apply_gradients(grads=grads_other)

    # Poincaré grads for embedding params
    def poincare_loss_fn(params_embed):
        return compute_poincare_loss(model, params_embed, state_other.params, batch, c)

    grads_embed = jax.grad(poincare_loss_fn)(state_embed.params)
    state_embed = state_embed.apply_gradients(grads=grads_embed)

    # Projection: keep embeddings inside Poincaré ball
    embed_params_unfrozen = unfreeze(state_embed.params)
    embed_matrix = embed_params_unfrozen["token_embed"]["embedding"]
    embed_params_unfrozen["token_embed"]["embedding"] = project_to_ball(embed_matrix, c)
    state_embed = state_embed.replace(params=freeze(embed_params_unfrozen))

    # Metrics for logging (no gradient effect)
    ce_val = compute_ce_loss(model, state_embed.params, state_other.params, batch)
    poin_val = compute_poincare_loss(model, state_embed.params, state_other.params, batch, c)
    total_loss = ce_val + lambda_poincare * poin_val

    metrics = {
        "loss_total": total_loss,
        "loss_ce": ce_val,
        "loss_poincare": poin_val
    }
    return state_embed, state_other, metrics


# -------------------------------------------------
# Eval step (no update)
# -------------------------------------------------
@partial(jax.jit, static_argnames=('model',))
def eval_step(state_embed, state_other, batch, c, lambda_poincare, model):
    """Validation step: loss computation only."""
    ce_val = compute_ce_loss(model, state_embed.params, state_other.params, batch)
    poin_val = compute_poincare_loss(model, state_embed.params, state_other.params, batch, c)
    total_loss = ce_val + lambda_poincare * poin_val
    return {
        "loss_total": total_loss,
        "loss_ce": ce_val,
        "loss_poincare": poin_val
    }

def eval_epoch(state_embed, state_other, val_data, batch_size, c, lambda_poincare, model,
               dataset_type="hf", max_batches=32):
    """
    Evaluate model over (part of) the validation dataset.
    """
    metrics_accum = {"loss_total": 0.0, "loss_ce": 0.0, "loss_poincare": 0.0}
    num_batches = 0

    if dataset_type == "pickle":
        input_ids = val_data["input_ids"]
        attention_mask = val_data["attention_mask"]
        total_samples = len(input_ids)
        num_samples = min(total_samples, batch_size * max_batches)
        indices = np.random.choice(total_samples, num_samples, replace=False)

        for start in range(0, num_samples, batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                "input_ids": jnp.array(input_ids[idx], dtype=jnp.int32),
                "attention_mask": jnp.array(attention_mask[idx], dtype=jnp.bool_)
            }
            batch_metrics = eval_step(state_embed, state_other, batch, c, lambda_poincare, model)
            for k in metrics_accum:
                metrics_accum[k] += float(batch_metrics[k])
            num_batches += 1

    else:  # HF dataset
        total_samples = len(val_data)
        num_samples = min(total_samples, batch_size * max_batches)
        indices = np.random.choice(total_samples, num_samples, replace=False)

        for start in range(0, num_samples, batch_size):
            idx = indices[start:start + batch_size]
            batch_data = val_data.select(idx)
            batch = {
                "input_ids": jnp.array(batch_data["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch_data["attention_mask"], dtype=jnp.bool_)
            }
            batch_metrics = eval_step(state_embed, state_other, batch, c, lambda_poincare, model)
            for k in metrics_accum:
                metrics_accum[k] += float(batch_metrics[k])
            num_batches += 1

    if num_batches:
        for k in metrics_accum:
            metrics_accum[k] /= num_batches
    else:
        for k in metrics_accum:
            metrics_accum[k] = float("inf")

    return metrics_accum
