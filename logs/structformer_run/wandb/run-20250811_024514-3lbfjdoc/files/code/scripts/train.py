"""
Train StructFormer + Poincaré on BabyLM-style data.

Adds dry-run ablation flags:
  --dry_run_structformer_only        -> training.mode = "structformer_only"
  --dry_run_structformer_poincare    -> training.mode = "structformer_poincare"

Dry runs also:
  - force num_epochs = 1 (unless already smaller)
  - clamp eval_interval_words to <= 200k for quick feedback

Final model still saved to 'main' by the training loop.
"""

import os
import argparse
import pickle
import logging
from types import SimpleNamespace
from typing import Any, Dict

import yaml
import jax
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from models.structformer_poincare import StructformerPoincare
from utils.logging_utils import create_training_logger
from utils.train_utils import train_structformer_poincare


# ---------------------------
# Small helpers
# ---------------------------
def _to_namespace(d: Dict[str, Any]) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(x) for x in d]
    return d

def _load_pickle_from_hub(repo_id: str, filename: str):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_datasets(cfg_ns):
    dtype = getattr(cfg_ns.data, "type", "hf")
    if dtype == "hf":
        ds = load_dataset(cfg_ns.data.hf_repo_id)
        train_split = getattr(cfg_ns.data, "train_split", "train")
        val_split = getattr(cfg_ns.data, "val_split", "validation")
        return ds[train_split], ds[val_split]
    elif dtype == "pickle":
        tr_repo = cfg_ns.data.train_tokenized_repo
        tr_file = cfg_ns.data.train_tokenized_file
        va_repo = cfg_ns.data.val_tokenized_repo
        va_file = cfg_ns.data.val_tokenized_file
        if not all([tr_repo, tr_file, va_repo, va_file]):
            raise ValueError("pickle mode requires train/val *_repo and *_file in config.data")
        return _load_pickle_from_hub(tr_repo, tr_file), _load_pickle_from_hub(va_repo, va_file)
    else:
        raise ValueError(f"Unknown data.type: {dtype}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Optional HF path like user/repo:branch")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional local run dir (logs, cache)")
    # dry-run ablations
    parser.add_argument("--dry_run_structformer_only", action="store_true",
                        help="Ablation: StructFormer only (Euclidean), skip Poincaré losses/updates")
    parser.add_argument("--dry_run_structformer_poincare", action="store_true",
                        help="Ablation: StructFormer + Poincaré (full pipeline) but short run")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = _to_namespace(cfg_dict)

    # Optional output directory (mainly for logs/artifacts)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # if you want W&B/tensorboard etc. to live here, make sure your logging.yaml points inside

    # Set dry-run mode if requested (flags override config)
    requested_mode = None
    if args.dry_run_structformer_only and args.dry_run_structformer_poincare:
        raise ValueError("Choose only one: --dry_run_structformer_only OR --dry_run_structformer_poincare")
    if args.dry_run_structformer_only:
        requested_mode = "structformer_only"
    elif args.dry_run_structformer_poincare:
        requested_mode = "structformer_poincare"

    # Ensure training namespace exists
    if not hasattr(cfg, "training"):
        cfg.training = SimpleNamespace()

    # Thread the mode into config (so train_utils can switch losses/updates)
    if requested_mode is not None:
        cfg.training.mode = requested_mode
    else:
        # default if not set in YAML
        if not hasattr(cfg.training, "mode"):
            cfg.training.mode = "structformer_poincare"

    # Shorten for dry runs (fast feedback): 1 epoch + tighter eval cadence
    if requested_mode is not None:
        current_epochs = int(getattr(cfg.training, "num_epochs", 1) or 1)
        cfg.training.num_epochs = min(current_epochs, 1)
        current_eval_words = int(getattr(cfg.training, "eval_interval_words", 1_000_000))
        cfg.training.eval_interval_words = min(current_eval_words, 200_000)

    # Seeding
    seed = int(getattr(cfg, "seed", 42))
    key = jax.random.PRNGKey(seed)

    # Tokenizer & pad_id
    tok = AutoTokenizer.from_pretrained(cfg.vocab_name)
    inferred_pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if inferred_pad is None and getattr(cfg, "pad_id", None) is None:
        raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set config.pad_id explicitly.")
    if getattr(cfg, "pad_id", None) is None:
        cfg.pad_id = inferred_pad

    # Build model (attention_input flag is read from cfg if present; default 'tangent')
    attention_input = getattr(cfg, "attention_input", "tangent")
    model = StructformerPoincare(
        vocab_size=tok.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        max_length=cfg.max_length,
        c=cfg.c,
        dropout_rate=cfg.dropout_rate,
        attention_input=attention_input,
    )

    # Logger (passes config for W&B enrichment)
    model_cfg_for_logging = SimpleNamespace(
        vocab_size=tok.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        max_length=cfg.max_length,
        c=cfg.c,
        training=getattr(cfg, "training", None),
    )
    logger = create_training_logger(cfg_dict, model_cfg_for_logging,
                                    max_words=int(getattr(cfg.training, "max_words", 100_000_000)))

    # Data
    train_ds, val_ds = _load_datasets(cfg)

    # Friendly startup log
    logging.getLogger(__name__).info(
        "Run mode: %s | epochs=%s | batch_size=%s | eval_every≈%s words | attention_input=%s",
        getattr(cfg.training, "mode", "structformer_poincare"),
        getattr(cfg.training, "num_epochs", None),
        getattr(cfg.training, "batch_size", None),
        getattr(cfg.training, "eval_interval_words", None),
        attention_input,
    )

    # Train (epochs-based; checkpoints/evals are word-based inside train_utils)
    _ = train_structformer_poincare(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_logger=logger,
        resume_from=args.resume,
        model_key=key,
    )

    logging.getLogger(__name__).info("Done.")


if __name__ == "__main__":
    main()
