"""
Train StructFormer (+ optional Poincaré) on BabyLM-style data.

Dry-run flags:
  --dry_run_structformer_only        -> training.mode = "structformer_only"
  --dry_run_structformer_poincare    -> training.mode = "structformer_poincare"

Dry runs also:
  - force num_epochs = 1 (if larger)
  - clamp eval_interval_words to <= 200k

Final model is saved to 'main' inside the training loop.
"""

import os
import argparse
import pickle
import logging
from typing import Any, Dict
from types import SimpleNamespace

import yaml
import jax
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from models.structformer_poincare import StructformerPoincare
from utils.logging_utils import create_training_logger
from utils.train_utils import train_structformer_poincare


# ---------------------------
# Utilities
# ---------------------------
def _to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(x) for x in d]
    return d

def _get(cfg: Any, path: str, default=None):
    """Safe getter for nested SimpleNamespace/dicts via dotted path (e.g., 'model.hidden_dim')."""
    cur = cfg
    for part in path.split('.'):
        if isinstance(cur, SimpleNamespace) and hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def _load_pickle_from_hub(repo_id: str, filename: str):
    """Download a pickle file from an HF dataset repo and load it."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Case 1: dict-of-lists → convert to list-of-dicts
    if isinstance(data, dict) and "input_ids" in data and "attention_mask" in data:
        data = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(data["input_ids"], data["attention_mask"])
        ]

    # Case 2: already list-of-dicts → nothing to do
    elif isinstance(data, list) and isinstance(data[0], dict):
        pass

    else:
        raise ValueError(f"Unexpected pickle format in {filename}: {type(data)}")

    return data

def _load_datasets(cfg_ns: Any):
    """
    Returns (train, val)
    - HF mode: returns Dataset objects
    - pickle mode: returns python lists or dicts as saved in your .pkl
    """
    dtype = _get(cfg_ns, "data.type", _get(cfg_ns, "data.dataset_type", "hf"))
    if dtype == "hf":
        hf_repo = _get(cfg_ns, "data.hf_repo_id")
        if not hf_repo:
            raise ValueError("In HF mode, data.hf_repo_id must be set.")
        ds = load_dataset(hf_repo)
        train_split = _get(cfg_ns, "data.train_split", "train")
        val_split = _get(cfg_ns, "data.val_split", "validation")
        return ds[train_split], ds[val_split]
    elif dtype == "pickle":
        tr_repo = _get(cfg_ns, "data.train_tokenized_repo")
        tr_file = _get(cfg_ns, "data.train_tokenized_file")
        va_repo = _get(cfg_ns, "data.val_tokenized_repo")
        va_file = _get(cfg_ns, "data.val_tokenized_file")
        if not all([tr_repo, tr_file, va_repo, va_file]):
            raise ValueError("pickle mode requires train/val *_repo and *_file in config.data")
        train_obj = _load_pickle_from_hub(tr_repo, tr_file)
        val_obj = _load_pickle_from_hub(va_repo, va_file)
        return train_obj, val_obj
    else:
        raise ValueError(f"Unknown data.type: {dtype}")

def _maybe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


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

    # Optional output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Dry-run mode resolution
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

    # Thread mode
    if requested_mode is not None:
        cfg.training.mode = requested_mode
    else:
        if not hasattr(cfg.training, "mode"):
            cfg.training.mode = "structformer_poincare"

    # Shorten for dry runs
    if requested_mode is not None:
        current_epochs = _maybe_int(_get(cfg, "training.num_epochs", 1), 1)
        cfg.training.num_epochs = min(current_epochs, 1)
        current_eval_words = _maybe_int(_get(cfg, "training.eval_interval_words", 1_000_000), 1_000_000)
        cfg.training.eval_interval_words = min(current_eval_words, 200_000)

    # Seeding
    seed = _maybe_int(_get(cfg, "seed", _get(cfg, "system.seed", 42)), 42)
    key = jax.random.PRNGKey(seed)

    # Tokenizer name: prefer data.vocab_name, then top-level vocab_name, then 'gpt2'
    vocab_name = _get(cfg, "data.vocab_name", _get(cfg, "vocab_name", "gpt2"))
    tok = AutoTokenizer.from_pretrained(vocab_name)
    # Infer pad id
    inferred_pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if inferred_pad is None and _get(cfg, "pad_id") is None:
        raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set config.pad_id explicitly.")
    if _get(cfg, "pad_id") is None:
        cfg.pad_id = inferred_pad

    # Model hyperparams (support nested 'model.*' or flat)
    hidden_dim  = _get(cfg, "model.hidden_dim",  _get(cfg, "hidden_dim", 256))
    num_layers  = _get(cfg, "model.num_layers",  _get(cfg, "num_layers", 8))
    num_heads   = _get(cfg, "model.num_heads",   _get(cfg, "num_heads", 8))
    max_length  = _get(cfg, "model.max_length",  _get(cfg, "max_length", 128))
    curvature_c = _get(cfg, "model.c",           _get(cfg, "c", 1.0))
    dropout     = _get(cfg, "model.dropout_rate", _get(cfg, "dropout_rate", 0.1))

    # Build model
    # Only pass 'attention_input' if the model actually accepts it
    attention_input = _get(cfg, "attention_input", None)
    try:
        if attention_input is None:
            model = StructformerPoincare(
                vocab_size=tok.vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_length=max_length,
                c=curvature_c,
                dropout_rate=dropout,
            )
        else:
            # Some forks may accept this kwarg
            model = StructformerPoincare(
                vocab_size=tok.vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_length=max_length,
                c=curvature_c,
                dropout_rate=dropout,
                attention_input=attention_input,  # try; will fail if model doesn't accept it
            )
    except TypeError:
        # Fallback: construct without that optional kwarg
        model = StructformerPoincare(
            vocab_size=tok.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            c=curvature_c,
            dropout_rate=dropout,
        )

    # Logger (W&B config enrichment)
    model_cfg_for_logging = SimpleNamespace(
        vocab_size=tok.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=max_length,
        c=curvature_c,
        training=getattr(cfg, "training", None),
    )
    logger = create_training_logger(
        cfg_dict,
        model_cfg_for_logging,
        max_words=int(_get(cfg, "training.max_words", 100_000_000)),
    )

    # Data
    train_ds, val_ds = _load_datasets(cfg)

    # Friendly startup log
    logging.getLogger(__name__).info(
        "Run mode: %s | epochs=%s | batch_size=%s | eval_every≈%s words | vocab=%s | pad_id=%s",
        getattr(cfg.training, "mode", "structformer_poincare"),
        _get(cfg, "training.num_epochs", None),
        _get(cfg, "training.batch_size", None),
        _get(cfg, "training.eval_interval_words", None),
        vocab_name,
        str(cfg.pad_id),
    )

    # Train (epochs-based; eval/checkpoints are word-based inside train_utils)
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

# python scripts/train.py \
#   --config configs/base.yaml \
#   --dry_run_structformer_only \
#   --output_dir runs/dry_structformer_only

# python scripts/train.py \
#   --config configs/base.yaml \
#   --dry_run_structformer_poincare \
#   --output_dir runs/dry_structformer_poincare

# python scripts/train.py \
#   --config configs/base.yaml \
#   --output_dir runs/full_structformer_poincare