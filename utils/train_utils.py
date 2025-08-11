"""
Training utilities for StructFormer + Poincare training
True single-forward-pass step with dual optimizers:
- CE loss updates non-embedding params
- Hyperbolic losses update embedding table (Riemannian grads + projection)
- Progress measured in 'words' = 0.75 * non-pad tokens
- Checkpoints: every 1M up to 10M, then every 10M up to 100M words
- Final model saved to Hugging Face 'main' branch
"""

import os
import sys
import time
from typing import Dict, Optional, Any, Tuple
from functools import partial
import logging

import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct
import optax

# Ensure the repository root is on the import path so that the sibling
# ``models`` package can be resolved when this module is imported as a
# top-level script (e.g. ``import train_utils`` from tests).
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.hyperbolic_geometry import (
    riemannian_gradient_conversion, poincare_proj, hyperbolic_diagnostics,
    poincare_distance_capped
)
from logging_utils import TrainingLogger
from save_utils import save_checkpoint_branch
from retrieve_utils import create_data_iterator

logger = logging.getLogger(__name__)

_BATCH_STATS: Dict[str, Any] = {}

# ----------------------------
# Training State Management
# ----------------------------
@struct.dataclass
class DualTrainState:
    """Two TrainStates: one for embeddings, one for other params."""
    embed_state: train_state.TrainState
    other_state: train_state.TrainState

    def get_merged_params(self) -> Dict[str, Any]:
        # embed_state.params is {'embed_table': ...}
        return {**self.other_state.params, **self.embed_state.params}

    def apply_gradients(self, embed_grads: Dict[str, Any], other_grads: Dict[str, Any]) -> "DualTrainState":
        new_embed_state = self.embed_state.apply_gradients(grads=embed_grads)
        new_other_state = self.other_state.apply_gradients(grads=other_grads)
        return DualTrainState(new_embed_state, new_other_state)


@struct.dataclass
class _TrainingCfg:
    lr_ce: float = 1e-3
    lr_riem: float = 1e-3
    lambda_h: float = 0.05
    lambda_tree: float = 0.05


@struct.dataclass
class _Config:
    c: float = 1.0
    training: _TrainingCfg = struct.field(default_factory=_TrainingCfg)


def create_dual_train_states(
    model,
    model_key: jax.random.PRNGKey,
    config,
    sample_input: Dict[str, jnp.ndarray]
) -> DualTrainState:
    """Initialize model and create two TrainStates with separate optax chains."""
    variables = model.init(model_key, **sample_input, training=True)
    params = variables['params']

    embed_params = {'embed_table': params['embed_table']}
    other_params = {k: v for k, v in params.items() if k != 'embed_table'}

    embed_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.training.lr_riem, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    other_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.training.lr_ce, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4),
    )

    embed_state = train_state.TrainState.create(apply_fn=model.apply, params=embed_params, tx=embed_optimizer)
    other_state = train_state.TrainState.create(apply_fn=model.apply, params=other_params, tx=other_optimizer)

    # Safety project initial embeddings
    safe_embed = poincare_proj(embed_params['embed_table'], config.c, eps_margin=1e-4)
    embed_state = embed_state.replace(params={'embed_table': safe_embed})

    logger.info("✅ Created dual training states")
    logger.info("   Embedding params: %s", f"{sum(p.size for p in jax.tree_util.tree_leaves(embed_params)):,}")
    logger.info("   Other params: %s", f"{sum(p.size for p in jax.tree_util.tree_leaves(other_params)):,}")
    return DualTrainState(embed_state, other_state)


# -----------------------------------------------------------------------------
# Convenience wrappers used in tests and simple scripts
# -----------------------------------------------------------------------------
def create_train_states(
    rng: jax.random.PRNGKey,
    model,
    vocab_size: int,
    seq_len: int,
    lr_ce: float = 1e-3,
    lr_riem: float = 1e-3,
    c: float = 1.0,
):
    """Initialize model parameters and optimizers for tests."""
    global _BATCH_STATS
    sample = {
        "input_ids": jnp.zeros((1, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((1, seq_len), dtype=jnp.bool_),
    }
    variables = model.init(rng, **sample, training=False)
    _BATCH_STATS = variables.get("batch_stats", {})
    params = variables["params"]

    embed_params = {"embed_table": params["embed_table"]}
    other_params = {k: v for k, v in params.items() if k != "embed_table"}

    embed_optimizer = optax.adamw(lr_riem)
    other_optimizer = optax.adamw(lr_ce)

    state_embed = train_state.TrainState.create(apply_fn=model.apply, params=embed_params, tx=embed_optimizer)
    state_other = train_state.TrainState.create(apply_fn=model.apply, params=other_params, tx=other_optimizer)

    safe_embed = poincare_proj(embed_params["embed_table"], c, eps_margin=1e-4)
    state_embed = state_embed.replace(params={"embed_table": safe_embed})

    return state_embed, state_other


def train_step(
    state_embed,
    state_other,
    batch: Dict[str, jnp.ndarray],
    c: float,
    lambda_poincare: float,
    model,
):
    """Single training step returning updated states and metrics."""
    params = {**state_other.params, **state_embed.params}

    def loss_fn(p):
        variables = {"params": p, "batch_stats": _BATCH_STATS}
        outs, new_bs = model.apply(
            variables,
            batch["input_ids"],
            batch.get("attention_mask"),
            training=False,
            mutable=["batch_stats"],
        )
        ce = compute_ce_loss(outs["logits"], batch["input_ids"], batch.get("attention_mask"))
        hyp = compute_local_hyperbolic_loss(outs["hyperbolic_embeds"], c)
        total = ce + lambda_poincare * hyp
        metrics = {"ce_loss": ce, "poincare_loss": hyp, "total_loss": total}
        return total, (metrics, new_bs["batch_stats"])

    (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    global _BATCH_STATS
    _BATCH_STATS = new_batch_stats

    embed_grads = {"embed_table": grads.get("embed_table", jnp.zeros_like(state_embed.params["embed_table"]))}
    other_grads = {k: v for k, v in grads.items() if k != "embed_table"}

    new_state_embed = state_embed.apply_gradients(grads=embed_grads)
    new_state_other = state_other.apply_gradients(grads=other_grads)

    safe_embed = poincare_proj(new_state_embed.params["embed_table"], c, eps_margin=1e-4)
    new_state_embed = new_state_embed.replace(params={"embed_table": safe_embed})

    return new_state_embed, new_state_other, metrics


def eval_step(
    state_embed,
    state_other,
    batch: Dict[str, jnp.ndarray],
    c: float,
    lambda_poincare: float,
    model,
):
    """Evaluation step wrapper matching the test harness signature."""
    dual = DualTrainState(state_embed, state_other)
    variables = {"params": dual.get_merged_params(), "batch_stats": _BATCH_STATS}
    outs = model.apply(variables, batch["input_ids"], batch.get("attention_mask"), training=False)
    ce_loss = compute_ce_loss(outs["logits"], batch["input_ids"], batch.get("attention_mask"))
    poincare_loss = compute_local_hyperbolic_loss(outs["hyperbolic_embeds"], c)
    perplexity = jnp.exp(ce_loss)
    return {
        "eval_ce_loss": ce_loss,
        "eval_poincare_loss": poincare_loss,
        "eval_perplexity": perplexity,
        "eval_total_loss": ce_loss + poincare_loss,
    }


def eval_epoch(
    state_embed,
    state_other,
    val_data,
    batch_size: int,
    c: float,
    lambda_poincare: float,
    model,
    dataset_type: str = "hf",
    max_batches: Optional[int] = None,
):
    """Evaluate over an entire epoch (or a limited number of batches)."""
    dual = DualTrainState(state_embed, state_other)
    config = _Config(c=c)

    total = {"eval_ce_loss": 0.0, "eval_poincare_loss": 0.0, "eval_total_loss": 0.0}
    count = 0

    if dataset_type == "hf":
        length = len(val_data)
        for start in range(0, length, batch_size):
            if max_batches is not None and count >= max_batches:
                break
            batch_slice = val_data[start : start + batch_size]
            batch = {
                "input_ids": jnp.array(batch_slice["input_ids"]),
                "attention_mask": jnp.array(batch_slice["attention_mask"]),
            }
            metrics = eval_step(state_embed, state_other, batch, c, lambda_poincare, model)
            for k in total:
                total[k] += float(metrics[k])
            count += 1
    else:
        batch = []
        for example in val_data:
            batch.append(example)
            if len(batch) == batch_size:
                if max_batches is not None and count >= max_batches:
                    break
                batch_arr = {
                    "input_ids": jnp.array([ex["input_ids"] for ex in batch]),
                    "attention_mask": jnp.array([ex["attention_mask"] for ex in batch]),
                }
                metrics = eval_step(state_embed, state_other, batch_arr, c, lambda_poincare, model)
                for k in total:
                    total[k] += float(metrics[k])
                count += 1
                batch = []
        if batch and (max_batches is None or count < max_batches):
            batch_arr = {
                "input_ids": jnp.array([ex["input_ids"] for ex in batch]),
                "attention_mask": jnp.array([ex["attention_mask"] for ex in batch]),
            }
            metrics = eval_step(state_embed, state_other, batch_arr, c, lambda_poincare, model)
            for k in total:
                total[k] += float(metrics[k])
            count += 1

    for k in total:
        total[k] = total[k] / max(count, 1)
    return total

# ----------------------------
# Losses
# ----------------------------
def compute_ce_loss(logits: jnp.ndarray, targets: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Causal LM CE over next-token positions."""
    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    ce = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, shifted_targets)  # [B, T-1]
    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        ce = ce * shifted_mask
        denom = jnp.maximum(jnp.sum(shifted_mask), 1.0)
        return jnp.sum(ce) / denom
    return jnp.mean(ce)

def compute_local_hyperbolic_loss(embeddings: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Average Poincaré distance between adjacent tokens."""
    cur = embeddings[:, :-1, :]
    nxt = embeddings[:, 1:, :]
    dists = jax.vmap(jax.vmap(lambda x1, x2: poincare_distance_capped(x1, x2, c, dmax=4.0)))(cur, nxt)
    return jnp.mean(dists)

def compute_tree_regularizer_loss(
    embeddings: jnp.ndarray,
    tree_distances: jnp.ndarray,
    sample_pairs: jnp.ndarray,
    c: float = 1.0,
    tau: float = 1.0
) -> jnp.ndarray:
    """(Optional) Encourage hyperbolic distances to correlate with given tree distances."""
    n = sample_pairs.shape[0]
    if n == 0:
        return jnp.array(0.0)

    def pair_loss(u_idx, v_idx):
        u = embeddings[u_idx]
        v = embeddings[v_idx]
        d_h = poincare_distance_capped(u, v, c, dmax=6.0)
        d_t = tree_distances[u_idx, v_idx]
        diff = d_h - tau * d_t
        return diff * diff

    def per_batch(batch_embeds):
        losses = jax.vmap(lambda ij: pair_loss(ij[0], ij[1]))(sample_pairs)
        return jnp.mean(losses)

    return jnp.mean(jax.vmap(per_batch)(embeddings))

# ----------------------------
# Single-pass training step (VJP-based)
# ----------------------------
@partial(jax.jit, static_argnames=('model', 'config'))
def train_step_single_pass(
    dual_state: DualTrainState,
    batch: Dict[str, jnp.ndarray],
    model,
    config,
    batch_stats: Optional[Dict[str, Any]] = None,
    tree_data: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[DualTrainState, Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    True single forward pass:
      1) Run forward once to get (logits, hyperbolic_embeds, new_batch_stats)
      2) Reuse a single VJP to get:
         - grads_other from d(CE)/dparams (apply only to non-embedding params)
         - grads_embed from d(HYP)/dparams (apply only to embedding table; convert to Riemannian)
      3) Update with two optimizers; project embeddings back into the ball.
    """
    if batch_stats is None:
        batch_stats = {}

    def fwd(params, inputs, attn_mask, mutable_stats):
        outs, new_mut = model.apply(
            {"params": params, "batch_stats": mutable_stats},
            inputs,
            attn_mask,
            training=True,
            mutable=["batch_stats"],
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        return (outs['logits'], outs['hyperbolic_embeds']), new_mut

    merged_params = {**dual_state.other_state.params, **dual_state.embed_state.params}

    # Forward once + VJP pullback
    (logits, hyp_embeds), new_mutable, pullback = jax.vjp(
        lambda p, x, m, bs: fwd(p, x, m, bs)[0],
        merged_params,
        batch['input_ids'],
        batch.get('attention_mask'),
        batch_stats,
        has_aux=True
    )

    # Compute losses
    ce = compute_ce_loss(logits, batch['input_ids'], batch.get('attention_mask'))
    local_hyp = compute_local_hyperbolic_loss(hyp_embeds, config.c)
    tree_loss = jnp.array(0.0)
    if tree_data is not None:
        tree_loss = compute_tree_regularizer_loss(
            hyp_embeds, tree_data['distances'], tree_data['sample_pairs'],
            c=config.c, tau=getattr(config.training, 'tau', 1.0)
        )
    lambda_h = getattr(config.training, 'lambda_h', 0.05)
    lambda_tree = getattr(config.training, 'lambda_tree', 0.05)
    hyp_total = lambda_h * local_hyp + lambda_tree * tree_loss
    total = ce + hyp_total

    # Build cotangents wrt (logits, hyp_embeds)
    def ce_wrt_logits(logits_):
        return compute_ce_loss(logits_, batch['input_ids'], batch.get('attention_mask'))

    def hyp_wrt_embeds(hyp_):
        local = compute_local_hyperbolic_loss(hyp_, config.c)
        return lambda_h * local + lambda_tree * tree_loss

    dCE_dlogits = jax.grad(ce_wrt_logits)(logits)
    dHYP_dembeds = jax.grad(hyp_wrt_embeds)(hyp_embeds)
    zero_logits = jax.tree_map(jnp.zeros_like, logits)
    zero_embeds = jax.tree_map(jnp.zeros_like, hyp_embeds)

    # Backprop once per branch, reusing the same pullback
    (grads_params_ce, _, _, _) = pullback((dCE_dlogits, zero_embeds))
    (grads_params_hyp, _, _, _) = pullback((zero_logits, dHYP_dembeds))

    # Split grads
    embed_grads_eu = {'embed_table': grads_params_hyp.get('embed_table', jnp.zeros_like(merged_params['embed_table']))}
    other_grads = {k: v for k, v in grads_params_ce.items() if k != 'embed_table'}

    # Riemannian conversion for embedding grads
    embed_params = dual_state.embed_state.params['embed_table']
    riem_grads = riemannian_gradient_conversion(embed_grads_eu['embed_table'], embed_params, config.c)
    embed_grads = {'embed_table': riem_grads}

    # Apply gradients
    new_state = dual_state.apply_gradients(embed_grads, other_grads)

    # Project embeddings to safe region
    safe_embed = poincare_proj(new_state.embed_state.params['embed_table'], config.c, eps_margin=1e-4)
    new_state = DualTrainState(
        new_state.embed_state.replace(params={'embed_table': safe_embed}),
        new_state.other_state
    )

    metrics = {
        'ce_loss': ce,
        'poincare_loss': local_hyp,
        'tree_loss': tree_loss,
        'hyperbolic_loss': hyp_total,
        'total_loss': total,
    }
    new_batch_stats = new_mutable.get('batch_stats', batch_stats)
    return new_state, metrics, new_batch_stats

# ----------------------------
# Eval
# ----------------------------
@partial(jax.jit, static_argnames=("model", "config"))
def _eval_step_internal(
    dual_state: DualTrainState,
    batch: Dict[str, jnp.ndarray],
    model,
    config,
    batch_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, jnp.ndarray]:
    """Internal JIT-able evaluation step used by higher level wrappers."""
    if batch_stats is None:
        batch_stats = {}
    merged_params = dual_state.get_merged_params()
    outs = model.apply(
        {"params": merged_params, "batch_stats": batch_stats},
        batch["input_ids"],
        batch.get("attention_mask"),
        training=False,
    )
    ce_loss = compute_ce_loss(outs["logits"], batch["input_ids"], batch.get("attention_mask"))
    poincare_loss = compute_local_hyperbolic_loss(outs["hyperbolic_embeds"], config.c)
    perplexity = jnp.exp(ce_loss)
    return {
        "eval_ce_loss": ce_loss,
        "eval_poincare_loss": poincare_loss,
        "eval_perplexity": perplexity,
        "eval_total_loss": ce_loss + poincare_loss,
    }

# ----------------------------
# Helpers: counting & milestones
# ----------------------------
def count_words_in_batch(batch: Dict[str, jnp.ndarray], pad_id: Optional[int]) -> float:
    """Words = 0.75 * non-pad tokens. Prefer attention_mask; fall back to input_ids != pad_id."""
    if 'attention_mask' in batch and batch['attention_mask'] is not None:
        nonpad = float(jnp.sum(batch['attention_mask']))
    else:
        if pad_id is None:
            # last resort: count all positions
            nonpad = float(jnp.prod(jnp.array(batch['input_ids'].shape)))
        else:
            nonpad = float(jnp.sum(batch['input_ids'] != pad_id))
    return 0.75 * nonpad

def build_word_milestones(max_words: int) -> Tuple[jnp.ndarray, int]:
    """[1M..10M] step 1M, then [20M..100M] step 10M, clipped to max_words."""
    first = [m * 1_000_000 for m in range(1, 11)]
    second = [m * 1_000_000 for m in range(20, 101, 10)]
    ms = [w for w in first + second if w <= max_words]
    return jnp.array(ms, dtype=jnp.int64), len(ms)

# ----------------------------
# Training loop
# ----------------------------
def run_training_loop(
    dual_state: DualTrainState,
    model,
    config,
    train_dataset,
    val_dataset,
    train_logger: TrainingLogger,
    batch_stats: Optional[Dict[str, Any]] = None,
    resume_step: int = 0,
    resume_words: float = 0.0
):
    """
    Epoch-based training loop:
      - Stop after `training.num_epochs` full passes over the dataset
      - Keep checkpoints/eval triggered by *word budget* (0.75 x non-pad tokens)
      - Word milestones: 1M..10M, then 20M..100M cumulative across epochs
      - Final model still saved to 'main'
    """
    if batch_stats is None:
        batch_stats = {}

    num_epochs = int(getattr(config.training, "num_epochs", 1))
    max_words = int(getattr(config.training, "max_words", 10_000_000))  # metadata only now
    batch_size = int(config.training.batch_size)
    eval_batch_size = int(getattr(config.training, 'eval_batch_size', batch_size))
    eval_interval_words = int(getattr(config.training, 'eval_interval_words', 1_000_000))
    pad_id = getattr(config, 'pad_id', None)

    # Milestones for checkpointing remain word-based
    milestones, _ = build_word_milestones(100_000_000)  # full set up to 100M
    # If you want to cap by planned total words across epochs, uncomment:
    # milestones, _ = build_word_milestones(num_epochs * max_words_estimate)
    next_idx = int(jnp.searchsorted(milestones, resume_words, side='right')) if milestones.size > 0 else 0

    step = int(resume_step)
    words_processed = float(resume_words)
    last_eval_words = float(resume_words)

    train_logger.logger.info("🚀 Starting training")
    train_logger.logger.info("   Epochs: %d", num_epochs)
    train_logger.logger.info("   Batch size: %d", batch_size)
    if milestones.size > 0:
        pretty = ", ".join(f"{int(m)//1_000_000}M" for m in list(milestones))
        train_logger.logger.info("   Checkpoint milestones (word-based): [%s]", pretty)
    else:
        train_logger.logger.info("   Checkpoint milestones: (none)")

    try:
        for epoch in range(num_epochs):
            train_logger.logger.info("🔁 Epoch %d/%d", epoch + 1, num_epochs)

            # Recreate the iterator each epoch (assumes it reshuffles when called anew)
            train_iterator = create_data_iterator(
                train_dataset,
                batch_size=batch_size,
                max_length=config.max_length,
                shuffle=True,
                drop_last=True
            )

            epoch_words = 0.0
            epoch_loss_acc = 0.0
            epoch_steps = 0

            for batch in train_iterator:
                t0 = time.time()
                dual_state, train_metrics, batch_stats = train_step_single_pass(
                    dual_state, batch, model, config, batch_stats
                )
                dt = max(1e-8, time.time() - t0)

                step += 1
                words_in_batch = count_words_in_batch(batch, pad_id)
                words_processed += words_in_batch
                epoch_words += words_in_batch
                epoch_steps += 1
                epoch_loss_acc += float(train_metrics['total_loss'])

                train_metrics['step_time'] = dt
                train_metrics['words_per_second'] = words_in_batch / dt

                # Hyperbolic diagnostics on the embedding table
                merged_params = dual_state.get_merged_params()
                if 'embed_table' in merged_params:
                    train_metrics.update(hyperbolic_diagnostics(merged_params['embed_table'], config.c))

                train_logger.log_training_step(
                    step=step,
                    words_processed=int(words_processed),
                    metrics=train_metrics,
                    words_in_batch=int(words_in_batch)
                )

                # Periodic eval by word budget
                if (words_processed - last_eval_words) >= eval_interval_words:
                    eval_metrics = run_evaluation(
                        dual_state, model, config, val_dataset, eval_batch_size, batch_stats, max_batches=100
                    )
                    train_logger.log_evaluation(step, int(words_processed), eval_metrics)
                    last_eval_words = words_processed

                # Word-based milestones for checkpointing (may cross multiple)
                while milestones.size > 0 and next_idx < milestones.size and words_processed >= float(milestones[next_idx]):
                    milestone_words = int(milestones[next_idx])
                    branch_name = f"{config.checkpointing.branch_prefix}_checkpoint_{milestone_words//1_000_000}M_words"

                    try:
                        save_checkpoint_branch(
                            params=merged_params,
                            config=config,
                            branch_name=branch_name,
                            repo_id=config.checkpointing.output_repo_id,
                            include_modeling_files=config.checkpointing.include_modeling_files,
                            model_file=config.checkpointing.model_file,
                            opt_state_embed=dual_state.embed_state.opt_state,
                            opt_state_other=dual_state.other_state.opt_state,
                            metrics=train_metrics,
                            step=step,
                            words_processed=milestone_words
                        )
                        train_logger.log_checkpoint_save(step, milestone_words, f"{config.checkpointing.output_repo_id}/{branch_name}")
                        train_logger.log_milestone(milestone_words, train_metrics)
                    except Exception as e:
                        logger.error("Failed to save checkpoint at milestone %s: %s", milestone_words, str(e))
                        raise

                    next_idx += 1

                if jnp.isnan(train_metrics['total_loss']):
                    train_logger.logger.error("🚨 NaN loss at step %s, stopping.", step)
                    raise RuntimeError("NaN loss")

            # End of epoch summary
            avg_epoch_loss = (epoch_loss_acc / max(epoch_steps, 1))
            train_logger.logger.info(
                "✅ Epoch %d/%d complete | avg loss: %.6f | words this epoch: %.0f | total words: %s",
                epoch + 1, num_epochs, avg_epoch_loss, epoch_words, f"{int(words_processed):,}"
            )

        # After all epochs: final eval + final save to MAIN
        train_logger.logger.info("🏁 All epochs completed — running final evaluation & save")
        final_eval = run_evaluation(
            dual_state, model, config, val_dataset, eval_batch_size, batch_stats, max_batches=None
        )
        train_logger.log_evaluation(step, int(words_processed), final_eval)

        try:
            branch_name = "main"  # final model to main
            merged_params = dual_state.get_merged_params()
            save_checkpoint_branch(
                params=merged_params,
                config=config,
                branch_name=branch_name,
                repo_id=config.checkpointing.output_repo_id,
                include_modeling_files=config.checkpointing.include_modeling_files,
                model_file=config.checkpointing.model_file,
                opt_state_embed=dual_state.embed_state.opt_state,
                opt_state_other=dual_state.other_state.opt_state,
                metrics=final_eval,
                step=step,
                words_processed=int(words_processed)
            )
            train_logger.log_checkpoint_save(step, int(words_processed), f"{config.checkpointing.output_repo_id}/{branch_name}")
            train_logger.log_milestone(int(words_processed), final_eval)
        except Exception as e:
            logger.error("Failed to save final checkpoint: %s", str(e))
            raise

    except KeyboardInterrupt:
        train_logger.logger.info("⚠️ Interrupted by Y!O!U!")
        try:
            branch_name = "interrupted_run"
            merged_params = dual_state.get_merged_params()
            save_checkpoint_branch(
                params=merged_params,
                config=config,
                branch_name=branch_name,
                repo_id=config.checkpointing.output_repo_id,
                include_modeling_files=config.checkpointing.include_modeling_files,
                model_file=config.checkpointing.model_file,
                opt_state_embed=dual_state.embed_state.opt_state,
                opt_state_other=dual_state.other_state.opt_state,
                metrics={},  # keep it light
                step=step,
                words_processed=int(words_processed)
            )
            train_logger.log_checkpoint_save(step, int(words_processed), f"{config.checkpointing.output_repo_id}/{branch_name}")
        except Exception:
            pass
        raise
    except Exception:
        train_logger.finalize()
        raise
    else:
        train_logger.finalize()

    return dual_state, int(words_processed)


# ----------------------------
# Eval loop
# ----------------------------
def run_evaluation(
    dual_state: DualTrainState,
    model,
    config,
    val_dataset,
    batch_size: int,
    batch_stats: Dict[str, Any],
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    eval_iterator = create_data_iterator(
        val_dataset, batch_size=batch_size, max_length=config.max_length, shuffle=False, drop_last=False
    )

    outs = []
    num_batches = 0
    for batch in eval_iterator:
        if max_batches is not None and num_batches >= max_batches:
            break
        outs.append(eval_step(dual_state, batch, model, config, batch_stats))
        num_batches += 1

    if not outs:
        return {}

    aggregated = {}
    keys = list(outs[0].keys())
    for k in keys:
        vals = jnp.array([float(o[k]) for o in outs], dtype=jnp.float32)
        aggregated[k] = float(jnp.mean(vals))
    logger.info("📊 Evaluated %d batches", num_batches)
    return aggregated

# ----------------------------
# Resume (HF Hub)
# ----------------------------
def load_checkpoint_and_resume(
    model,
    config,
    checkpoint_path: str,
    sample_input: Dict[str, jnp.ndarray]
) -> Tuple[DualTrainState, int, float, Dict[str, Any]]:
    """Load from HF Hub (repo/branch) and reconstruct DualTrainState."""
    from .save_utils import load_checkpoint_from_hub

    if '/' in checkpoint_path:
        parts = checkpoint_path.split('/')
        repo_id = '/'.join(parts[:-1])
        branch_name = parts[-1]
        checkpoint_data = load_checkpoint_from_hub(repo_id=repo_id, branch_name=branch_name, load_optimizer_states=True)
    else:
        raise NotImplementedError("Local checkpoint loading not implemented")

    if checkpoint_data is None:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    params = checkpoint_data['params']
    opt_state_embed = checkpoint_data.get('opt_state_embed')
    opt_state_other = checkpoint_data.get('opt_state_other')
    metadata = checkpoint_data.get('metadata', {})

    resume_step = int(metadata.get('step', 0))
    resume_words = float(metadata.get('words_processed', 0))

    embed_params = {'embed_table': params['embed_table']}
    other_params = {k: v for k, v in params.items() if k != 'embed_table'}

    if opt_state_embed is None or opt_state_other is None:
        logger.warning("Optimizer states missing; creating fresh optimizers")
        dual_state = create_dual_train_states(model, random.PRNGKey(0), config, sample_input)
        dual_state = DualTrainState(
            dual_state.embed_state.replace(params=embed_params),
            dual_state.other_state.replace(params=other_params)
        )
    else:
        embed_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=config.training.lr_riem, weight_decay=0.0),
        )
        other_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=config.training.lr_ce, weight_decay=1e-4),
        )

        embed_state = train_state.TrainState(
            step=resume_step,
            apply_fn=model.apply,
            params=embed_params,
            tx=embed_optimizer,
            opt_state=opt_state_embed,
        )
        other_state = train_state.TrainState(
            step=resume_step,
            apply_fn=model.apply,
            params=other_params,
            tx=other_optimizer,
            opt_state=opt_state_other,
        )
        dual_state = DualTrainState(embed_state, other_state)

    batch_stats = checkpoint_data.get('batch_stats', {})
    logger.info("✅ Checkpoint loaded: step %s, words %s", f"{resume_step:,}", f"{int(resume_words):,}")
    return dual_state, resume_step, resume_words, batch_stats

# ----------------------------
# Misc helpers for demos
# ----------------------------
def create_sample_tree_data(batch_size: int, seq_length: int, num_pairs: int = 50) -> Dict[str, jnp.ndarray]:
    """Dummy tree distances + sampled index pairs."""
    key = random.PRNGKey(42)
    indices = random.randint(key, (num_pairs, 2), 0, seq_length)
    tree_distances = jnp.ones((seq_length, seq_length))  # placeholder
    return {'sample_pairs': indices, 'distances': tree_distances}

# ----------------------------
# Orchestrator
# ----------------------------
def train_structformer_poincare(
    model,
    config,
    train_dataset,
    val_dataset,
    train_logger: TrainingLogger,
    resume_from: Optional[str] = None,
    model_key: Optional[jax.random.PRNGKey] = None
) -> DualTrainState:
    """Init/resume → train → final eval + checkpoint."""
    if model_key is None:
        model_key = random.PRNGKey(42)

    sample_input = {
        'input_ids': jnp.ones((1, config.max_length), dtype=jnp.int32),
        'attention_mask': jnp.ones((1, config.max_length), dtype=jnp.float32),
    }

    if resume_from is not None:
        dual_state, resume_step, resume_words, batch_stats = load_checkpoint_and_resume(
            model, config, resume_from, sample_input
        )
        train_logger.logger.info("🔄 Resuming from %s", resume_from)
    else:
        dual_state = create_dual_train_states(model, model_key, config, sample_input)
        resume_step, resume_words = 0, 0.0
        batch_stats = {}
        train_logger.logger.info("🆕 Starting fresh training")

    from .logging_utils import log_model_summary
    merged_params = dual_state.get_merged_params()
    log_model_summary(train_logger, model, merged_params, sample_input)

    final_dual_state, final_words = run_training_loop(
        dual_state=dual_state,
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_logger=train_logger,
        batch_stats=batch_stats,
        resume_step=resume_step,
        resume_words=resume_words
    )
    train_logger.logger.info("🎉 Training completed! Final words processed: %s", f"{final_words:,}")
    return final_dual_state
