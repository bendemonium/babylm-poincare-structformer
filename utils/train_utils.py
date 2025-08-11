"""
Training utilities for StructFormer + Poincaré training

Design:
- True single-forward-pass step with dual optimizers
  * CE loss updates non-embedding params
  * Hyperbolic losses update embedding table (Riemannian grads + projection)
- Progress measured in 'words' = 0.75 * non-pad tokens
- Checkpoints: every 1M up to 10M, then every 10M up to 100M words
- Final model saved to Hugging Face 'main' branch
"""

import time
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct
import optax

from .hyperbolic_geometry import (
    riemannian_gradient_conversion,
    poincare_proj,
    hyperbolic_diagnostics,
    poincare_distance_capped,
)
from .logging_utils import TrainingLogger
from .save_utils import save_checkpoint_branch
from .retrieve_utils import create_data_iterator

logger = logging.getLogger(__name__)

# Global (Python) holder for mutable batch stats (for models that use it)
_BATCH_STATS: Dict[str, Any] = {}

# ----------------------------
# Training state
# ----------------------------
@struct.dataclass
class DualTrainState:
    """Two TrainStates: one for embeddings, one for other params."""
    embed_state: train_state.TrainState
    other_state: train_state.TrainState

    def get_merged_params(self) -> Dict[str, Any]:
        # embed_state.params is {'embed_table': ...}
        return {**self.other_state.params, **self.embed_state.params}

    def apply_gradients(
        self,
        embed_grads: Dict[str, Any],
        other_grads: Dict[str, Any],
    ) -> "DualTrainState":
        new_embed_state = self.embed_state.apply_gradients(grads=embed_grads)
        new_other_state = self.other_state.apply_gradients(grads=other_grads)
        return DualTrainState(new_embed_state, new_other_state)


def create_dual_train_states(
    model,
    model_key: jax.random.PRNGKey,
    config,
    sample_input: Dict[str, jnp.ndarray],
) -> DualTrainState:
    """Initialize model and create two TrainStates with separate optax chains."""
    variables = model.init(model_key, **sample_input, training=True)
    params = variables["params"]

    # Split params
    embed_params = {"embed_table": params["embed_table"]}
    other_params = {k: v for k, v in params.items() if k != "embed_table"}

    # Optims
    embed_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=float(config.training.lr_riem),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        ),
    )
    other_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=float(config.training.lr_ce),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-4,
        ),
    )

    embed_state = train_state.TrainState.create(
        apply_fn=model.apply, params=embed_params, tx=embed_optimizer
    )
    other_state = train_state.TrainState.create(
        apply_fn=model.apply, params=other_params, tx=other_optimizer
    )

    # Safety project initial embeddings
    safe_embed = poincare_proj(embed_params["embed_table"], float(config.c), eps_margin=1e-4)
    embed_state = embed_state.replace(params={"embed_table": safe_embed})

    logger.info("✅ Created dual training states")
    logger.info(
        "   Embedding params: %s",
        f"{sum(p.size for p in jax.tree_util.tree_leaves(embed_params)):,}",
    )
    logger.info(
        "   Other params: %s",
        f"{sum(p.size for p in jax.tree_util.tree_leaves(other_params)):,}",
    )
    return DualTrainState(embed_state, other_state)

# ----------------------------
# Losses
# ----------------------------
def compute_ce_loss(
    logits: jnp.ndarray, targets: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
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
    tau: float = 1.0,
) -> jnp.ndarray:
    """(Optional) Encourage hyperbolic distances to correlate with supplied tree distances."""
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
@partial(jax.jit, static_argnames=("model",))
def train_step_single_pass(
    dual_state: DualTrainState,
    batch: Dict[str, jnp.ndarray],
    model,
    c: float,
    lambda_h: float,
    lambda_tree: float,
    tau: float,
    batch_stats: Optional[Dict[str, Any]] = None,
    tree_data: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[DualTrainState, Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    True single forward pass:
      1) Run forward once to get (logits, hyperbolic_embeds, new_batch_stats)
      2) Use one VJP and provide cotangents for two heads:
         - grads_other from d(CE)/dparams (apply only to non-embedding params)
         - grads_embed from d(HYP)/dparams (apply only to embedding table; convert to Riemannian)
      3) Update with two optimizers; project embeddings back into the ball.
    """
    if batch_stats is None:
        batch_stats = {}

    def fwd(params, inputs, attn_mask, mutable_stats):
        (outs, new_mut) = model.apply(
            {"params": params, "batch_stats": mutable_stats},
            inputs,
            attn_mask,
            training=True,
            mutable=["batch_stats"],
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        # Return both heads we’ll differentiate wrt
        return (outs["logits"], outs["hyperbolic_embeds"]), new_mut

    merged_params = {**dual_state.other_state.params, **dual_state.embed_state.params}

    # Run forward once, get VJP pullback
    ((logits, hyp_embeds), new_mutable), pullback = jax.vjp(
        fwd,
        merged_params,
        batch["input_ids"],
        batch.get("attention_mask"),
        batch_stats,
        has_aux=True,
    )

    # Compute scalar losses
    ce = compute_ce_loss(logits, batch["input_ids"], batch.get("attention_mask"))
    local_hyp = compute_local_hyperbolic_loss(hyp_embeds, c)
    tree_loss = jnp.array(0.0)
    if tree_data is not None:
        tree_loss = compute_tree_regularizer_loss(
            hyp_embeds, tree_data["distances"], tree_data["sample_pairs"], c=c, tau=tau
        )
    hyp_total = lambda_h * local_hyp + lambda_tree * tree_loss
    total = ce + hyp_total

    # Build cotangents wrt outputs (logits, hyp_embeds)
    def ce_wrt_logits(lg):
        return compute_ce_loss(lg, batch["input_ids"], batch.get("attention_mask"))

    def hyp_wrt_embeds(hy):
        local = compute_local_hyperbolic_loss(hy, c)
        return lambda_h * local + lambda_tree * tree_loss

    dCE_dlogits = jax.grad(ce_wrt_logits)(logits)
    dHYP_dembeds = jax.grad(hyp_wrt_embeds)(hyp_embeds)
    zero_logits = jax.tree_map(jnp.zeros_like, logits)
    zero_embeds = jax.tree_map(jnp.zeros_like, hyp_embeds)

    # Pull back cotangents to param grads (params is the first arg)
    grads_params_ce, _, _, _ = pullback((dCE_dlogits, zero_embeds))
    grads_params_hyp, _, _, _ = pullback((zero_logits, dHYP_dembeds))

    # Split grads
    embed_grads_eu = {
        "embed_table": grads_params_hyp.get(
            "embed_table", jnp.zeros_like(merged_params["embed_table"])
        )
    }
    other_grads = {k: v for k, v in grads_params_ce.items() if k != "embed_table"}

    # Riemannian conversion for embedding grads
    embed_params = dual_state.embed_state.params["embed_table"]
    riem_grads = riemannian_gradient_conversion(embed_grads_eu["embed_table"], embed_params, c)
    embed_grads = {"embed_table": riem_grads}

    # Apply gradients
    new_state = dual_state.apply_gradients(embed_grads, other_grads)

    # Project embeddings back into the ball
    safe_embed = poincare_proj(new_state.embed_state.params["embed_table"], c, eps_margin=1e-4)
    new_state = DualTrainState(
        new_state.embed_state.replace(params={"embed_table": safe_embed}),
        new_state.other_state,
    )

    metrics = {
        "ce_loss": ce,
        "poincare_loss": local_hyp,
        "tree_loss": tree_loss,
        "hyperbolic_loss": hyp_total,
        "total_loss": total,
    }
    new_batch_stats = new_mutable.get("batch_stats", batch_stats)
    return new_state, metrics, new_batch_stats

# ----------------------------
# Eval
# ----------------------------
@partial(jax.jit, static_argnames=("model",))
def _eval_step_internal(
    dual_state: DualTrainState,
    batch: Dict[str, jnp.ndarray],
    model,
    c: float,
    batch_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, jnp.ndarray]:
    """Eval: one forward, CE + local hyperbolic metrics (no grads)."""
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
    poincare_loss = compute_local_hyperbolic_loss(outs["hyperbolic_embeds"], c)
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
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        nonpad = float(jnp.sum(batch["attention_mask"]))
    else:
        if pad_id is None:
            nonpad = float(jnp.prod(jnp.array(batch["input_ids"].shape)))
        else:
            nonpad = float(jnp.sum(batch["input_ids"] != pad_id))
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
    resume_words: float = 0.0,
):
    """
    Epoch-based training loop:
      - Stop after `training.num_epochs` full passes over the dataset
      - Keep eval & checkpoint triggers word-based (0.75 x non-pad tokens)
      - Word milestones: 1M..10M, then 20M..100M cumulative across epochs
      - Final model saved to 'main'
    """
    if batch_stats is None:
        batch_stats = {}

    num_epochs = int(getattr(config.training, "num_epochs", 1))
    batch_size = int(config.training.batch_size)
    eval_batch_size = int(getattr(config.training, "eval_batch_size", batch_size))
    eval_interval_words = int(getattr(config.training, "eval_interval_words", 1_000_000))
    pad_id = getattr(config, "pad_id", None)

    # Checkpoint milestones (up to 100M)
    milestones, _ = build_word_milestones(100_000_000)
    next_idx = int(jnp.searchsorted(milestones, resume_words, side="right")) if milestones.size > 0 else 0

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

            # Fresh iterator (assumed to reshuffle)
            train_iterator = create_data_iterator(
                train_dataset,
                batch_size=batch_size,
                max_length=config.max_length,
                shuffle=True,
                drop_last=True,
            )

            epoch_words = 0.0
            epoch_loss_acc = 0.0
            epoch_steps = 0

            for batch in train_iterator:
                t0 = time.time()
                # Pass only scalars to jitted step (no SimpleNamespace).
                dual_state, train_metrics, batch_stats = train_step_single_pass(
                    dual_state=dual_state,
                    batch=batch,
                    model=model,
                    c=float(config.c),
                    lambda_h=float(getattr(config.training, "lambda_h", 0.05)),
                    lambda_tree=float(getattr(config.training, "lambda_tree", 0.05)),
                    tau=float(getattr(config.training, "tau", 1.0)),
                    batch_stats=batch_stats,
                    tree_data=None,
                )
                dt = max(1e-8, time.time() - t0)

                step += 1
                words_in_batch = count_words_in_batch(batch, pad_id)
                words_processed += words_in_batch
                epoch_words += words_in_batch
                epoch_steps += 1
                epoch_loss_acc += float(train_metrics["total_loss"])

                train_metrics["step_time"] = dt
                train_metrics["words_per_second"] = words_in_batch / dt

                # Hyperbolic diagnostics on the embedding table (cheap summary)
                merged_params = dual_state.get_merged_params()
                if "embed_table" in merged_params:
                    train_metrics.update(hyperbolic_diagnostics(merged_params["embed_table"], float(config.c)))

                train_logger.log_training_step(
                    step=step,
                    words_processed=int(words_processed),
                    metrics=train_metrics,
                    words_in_batch=int(words_in_batch),
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
                            words_processed=milestone_words,
                        )
                        train_logger.log_checkpoint_save(
                            step, milestone_words, f"{config.checkpointing.output_repo_id}/{branch_name}"
                        )
                        train_logger.log_milestone(milestone_words, train_metrics)
                    except Exception as e:
                        logger.error("Failed to save checkpoint at milestone %s: %s", milestone_words, str(e))
                        raise

                    next_idx += 1

                if jnp.isnan(train_metrics["total_loss"]):
                    train_logger.logger.error("🚨 NaN loss at step %s, stopping.", step)
                    raise RuntimeError("NaN loss")

            # End of epoch summary
            avg_epoch_loss = (epoch_loss_acc / max(epoch_steps, 1))
            train_logger.logger.info(
                "✅ Epoch %d/%d complete | avg loss: %.6f | words this epoch: %.0f | total words: %s",
                epoch + 1,
                num_epochs,
                avg_epoch_loss,
                epoch_words,
                f"{int(words_processed):,}",
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
                words_processed=int(words_processed),
            )
            train_logger.log_checkpoint_save(
                step, int(words_processed), f"{config.checkpointing.output_repo_id}/{branch_name}"
            )
            train_logger.log_milestone(int(words_processed), final_eval)
        except Exception as e:
            logger.error("Failed to save final checkpoint: %s", str(e))
            raise

    except KeyboardInterrupt:
        train_logger.logger.info("⚠️ Training interrupted by user")
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
                words_processed=int(words_processed),
            )
            train_logger.log_checkpoint_save(
                step, int(words_processed), f"{config.checkpointing.output_repo_id}/{branch_name}"
            )
        except Exception:
            pass
        raise
    finally:
        train_logger.finalize()

    return dual_state, int(words_processed)

# ----------------------------
# Eval loop (driver)
# ----------------------------
def run_evaluation(
    dual_state: DualTrainState,
    model,
    config,
    val_dataset,
    batch_size: int,
    batch_stats: Dict[str, Any],
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    eval_iterator = create_data_iterator(
        val_dataset,
        batch_size=batch_size,
        max_length=config.max_length,
        shuffle=False,
        drop_last=False,
    )

    outs = []
    num_batches = 0
    for batch in eval_iterator:
        if max_batches is not None and num_batches >= max_batches:
            break
        outs.append(
            _eval_step_internal(
                dual_state=dual_state,
                batch=batch,
                model=model,
                c=float(config.c),
                batch_stats=batch_stats,
            )
        )
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
    sample_input: Dict[str, jnp.ndarray],
) -> Tuple[DualTrainState, int, float, Dict[str, Any]]:
    """Load from HF Hub (repo/branch) and reconstruct DualTrainState."""
    from .save_utils import load_checkpoint_from_hub

    if "/" in checkpoint_path:
        parts = checkpoint_path.split("/")
        repo_id = "/".join(parts[:-1])
        branch_name = parts[-1]
        checkpoint_data = load_checkpoint_from_hub(
            repo_id=repo_id, branch_name=branch_name, load_optimizer_states=True
        )
    else:
        raise NotImplementedError("Local checkpoint loading not implemented")

    if checkpoint_data is None:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    params = checkpoint_data["params"]
    opt_state_embed = checkpoint_data.get("opt_state_embed")
    opt_state_other = checkpoint_data.get("opt_state_other")
    metadata = checkpoint_data.get("metadata", {})

    resume_step = int(metadata.get("step", 0))
    resume_words = float(metadata.get("words_processed", 0))

    embed_params = {"embed_table": params["embed_table"]}
    other_params = {k: v for k, v in params.items() if k != "embed_table"}

    if opt_state_embed is None or opt_state_other is None:
        logger.warning("Optimizer states missing; creating fresh optimizers")
        dual_state = create_dual_train_states(model, random.PRNGKey(0), config, sample_input)
        dual_state = DualTrainState(
            dual_state.embed_state.replace(params=embed_params),
            dual_state.other_state.replace(params=other_params),
        )
    else:
        embed_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=float(config.training.lr_riem), weight_decay=0.0),
        )
        other_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=float(config.training.lr_ce), weight_decay=1e-4),
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

    batch_stats = checkpoint_data.get("batch_stats", {})
    logger.info(
        "✅ Checkpoint loaded: step %s, words %s", f"{resume_step:,}", f"{int(resume_words):,}"
    )
    return dual_state, resume_step, resume_words, batch_stats

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
    model_key: Optional[jax.random.PRNGKey] = None,
) -> DualTrainState:
    """Init/resume → train → final eval + checkpoint."""
    if model_key is None:
        model_key = random.PRNGKey(42)

    sample_input = {
        "input_ids": jnp.ones((1, config.max_length), dtype=jnp.int32),
        "attention_mask": jnp.ones((1, config.max_length), dtype=jnp.float32),
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

    # Quick model summary
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
        resume_words=resume_words,
    )
    train_logger.logger.info(
        "🎉 Training completed! Final words processed: %s", f"{final_words:,}"
    )
    return final_dual_state