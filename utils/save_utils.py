"""
Save utilities for StructFormer + Poincaré training
Handles checkpointing, HuggingFace Hub uploads, and model serialization
"""

import os
import json
import shutil
import tempfile
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.serialization import to_bytes, from_bytes
from huggingface_hub import HfApi, create_repo, upload_folder

from models.hyperbolic_geometry import hyperbolic_diagnostics

logger = logging.getLogger(__name__)

# ----------------------------
# Helpers
# ----------------------------
def _now_iso() -> str:
    # jnp.datetime64('now') is unreliable; use real clock
    return datetime.now(timezone.utc).isoformat()

def _is_jax_array(x: Any) -> bool:
    # JAX 0.6.x unified arrays
    return hasattr(x, "__class__") and x.__class__.__name__ in {"Array", "DeviceArray"}

def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    """
    if hasattr(obj, "__dict__"):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
    if hasattr(obj, "_asdict"):
        return {k: make_json_serializable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if _is_jax_array(obj):
        try:
            # Don’t explode JSON with huge arrays
            return obj.tolist() if obj.size < 100 else f"JAX array shape {tuple(obj.shape)}"
        except Exception:
            return "JAX array"
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

def sanitize(obj: Any) -> Any:
    """Sanitize objects (e.g. JAX arrays) for JSON serialization."""
    return make_json_serializable(obj)

# ----------------------------
# Config & metadata
# ----------------------------
class CheckpointConfig:
    """Configuration for checkpointing and saving."""
    def __init__(self, config_dict):
        self.checkpointing = config_dict.get("checkpointing", {})
        self.use_word_milestones = self.checkpointing.get("use_word_milestones", True)
        self.max_words = self.checkpointing.get("max_words", 100_000_000)
        self.output_repo_id = self.checkpointing.get("output_repo_id")
        self.branch_prefix = self.checkpointing.get("branch_prefix", "checkpoint")
        self.include_modeling_files = self.checkpointing.get("include_modeling_files", [])
        self.model_file = self.checkpointing.get("model_file", "models/structformer_poincare")

        self.logging = config_dict.get("logging", {})
        self.log_dir = self.logging.get("log_dir", "logs/structformer_run")

def write_config(config, save_dir: str, model_file: Optional[str] = None):
    """
    Write configuration to JSON (with metadata).
    """
    os.makedirs(save_dir, exist_ok=True)
    cfg = make_json_serializable(config)
    cfg["_metadata"] = {
        "model_type": "structformer_poincare",
        "framework": "jax_flax",
        "model_file": model_file,
        "created_at": _now_iso(),
    }
    path = os.path.join(save_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Config saved to {path}")

# ----------------------------
# Local checkpoints (Flax)
# ----------------------------
def save_flax_checkpoint(
    params: Dict,
    opt_state_embed: Any,
    opt_state_other: Any,
    step: int,
    save_dir: str,
    prefix: str = "checkpoint",
):
    """
    Save a checkpoint containing params and both optimizer states.
    """
    os.makedirs(save_dir, exist_ok=True)
    target = {
        "params": params,
        "opt_state_embed": opt_state_embed,
        "opt_state_other": opt_state_other,
        "step": int(step),
        "framework": "jax_flax",
        "model_type": "structformer_poincare",
    }
    checkpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=target,
        step=step,
        prefix=prefix,
        keep=5,
        overwrite=True,
    )
    logger.info(f"✅ Flax checkpoint saved: {save_dir}/{prefix}_{step}")

def load_flax_checkpoint(
    ckpt_dir: str,
    step: Optional[int] = None,
    prefix: str = "checkpoint",
) -> Optional[Dict]:
    """
    Load a checkpoint (latest if step is None).
    """
    if not os.path.exists(ckpt_dir):
        logger.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return None
    data = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None,
        step=step,
        prefix=prefix,
    )
    if data is None:
        logger.warning(f"No checkpoint found in {ckpt_dir}")
        return None
    logger.info(f"✅ Loaded checkpoint from {ckpt_dir}, step {data.get('step', 'unknown')}")
    return data

def save_training_state(
    params: Dict,
    opt_state_embed: Any,
    opt_state_other: Any,
    metrics: Dict,
    config: Any,
    step: int,
    words_processed: int,
    save_dir: str,
):
    """
    Save checkpoint + metrics/config snapshot.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Checkpoint
    save_flax_checkpoint(params, opt_state_embed, opt_state_other, step, save_dir, prefix="training_state")

    # 2) Training info (metrics & counters)
    training_info = {
        "step": int(step),
        "words_processed": int(words_processed),
        "metrics": make_json_serializable(metrics),
        "timestamp": _now_iso(),
    }
    info_path = os.path.join(save_dir, "training_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    # 3) Config
    write_config(config, save_dir)
    logger.info(f"✅ Training state saved to {save_dir}")

# ----------------------------
# Hugging Face Hub
# ----------------------------
def create_huggingface_repo(repo_id: str, private: bool = True, repo_type: str = "model", create_main_branch: bool = True):
    """
    Create HF repo if it doesn't exist.
    """
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type=repo_type)
        logger.info(f"Repository {repo_id} already exists")
        return
    except Exception:
        pass
    create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
    if create_main_branch:
        # Optional: ensure main branch exists (usually created automatically)
        try:
            api.create_branch(repo_id, branch="main", repo_type=repo_type)
        except Exception:
            pass
    logger.info(f"✅ Created HuggingFace repository: {repo_id}")

def copy_modeling_files(source_files: List[str], target_dir: str):
    """
    Copy selected modeling files into upload folder.
    """
    for src in source_files or []:
        if not os.path.exists(src):
            logger.warning(f"Modeling file not found: {src}")
            continue
        rel = os.path.normpath(src)  # preserve relative structure as given
        dst = os.path.join(target_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} → {dst}")

def save_checkpoint_branch(
    params: Dict,
    config: Any,
    branch_name: str,
    repo_id: str,
    include_modeling_files: Optional[List[str]] = None,
    model_file: Optional[str] = None,
    opt_state_embed: Any = None,
    opt_state_other: Any = None,
    metrics: Dict = None,
    step: int = 0,
    words_processed: int = 0,
):
    """
    Package and upload a checkpoint to a specific Hub branch.
    """
    logger.info(f"Saving checkpoint to branch {branch_name}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1) Ensure repo
        create_huggingface_repo(repo_id, private=True, repo_type="model")

        # 2) Params
        params_path = os.path.join(tmp_dir, "model_params.flax")
        with open(params_path, "wb") as f:
            f.write(to_bytes(params))

        # 3) Opt states
        if opt_state_embed is not None:
            with open(os.path.join(tmp_dir, "opt_state_embed.flax"), "wb") as f:
                f.write(to_bytes(opt_state_embed))
        if opt_state_other is not None:
            with open(os.path.join(tmp_dir, "opt_state_other.flax"), "wb") as f:
                f.write(to_bytes(opt_state_other))

        # 4) Config
        write_config(config, tmp_dir, model_file)

        # 5) Metadata (optionally include diagnostics on the *embedding table*)
        metadata = {
            "step": int(step),
            "words_processed": int(words_processed),
            "branch_name": branch_name,
            "model_type": "structformer_poincare",
            "framework": "jax_flax",
            "timestamp": _now_iso(),
        }
        try:
            # If params contain the embedding table, log geometry stats
            embed = None
            # Common Flax param structure: params['embed_table'] or params['params']['embed_table']
            if isinstance(params, dict) and "embed_table" in params:
                embed = params["embed_table"]
            elif isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
                if "embed_table" in params["params"]:
                    embed = params["params"]["embed_table"]
            if embed is not None:
                metadata["hyperbolic_diagnostics"] = make_json_serializable(
                    hyperbolic_diagnostics(embed, getattr(config, "c", 1.0))
                )
        except Exception:
            pass

        with open(os.path.join(tmp_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 6) Modeling files
        if include_modeling_files:
            copy_modeling_files(include_modeling_files, tmp_dir)

        # 7) README
        readme = f"""# StructFormer + Poincaré Checkpoint

Checkpoint from training StructFormer with Poincaré (hyperbolic) embeddings.

## Details
- **Framework**: JAX/Flax
- **Model Type**: StructFormer + Poincaré
- **Training Step**: {step:,}
- **Words Processed**: {words_processed:,}
- **Branch**: {branch_name}
- **Timestamp**: {metadata['timestamp']}

## Files
- `model_params.flax`
- `opt_state_embed.flax` (optional)
- `opt_state_other.flax` (optional)
- `config.json`
- `training_metadata.json`
- modeling code (if included)
"""
        with open(os.path.join(tmp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

        # 8) Upload
        logger.info(f"Uploading to {repo_id}@{branch_name} ...")
        upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=branch_name,        # will create the branch if it doesn't exist
            commit_message=f"Checkpoint at {words_processed:,} words (step {step:,})",
            create_pr=False,
        )

    logger.info(f"✅ Checkpoint saved to {repo_id}/{branch_name}")

def load_checkpoint_from_hub(
    repo_id: str,
    branch_name: str = "main",
    load_optimizer_states: bool = True,
) -> Optional[Dict]:
    """
    Download a branch snapshot and read params/opt states/metadata/config.
    """
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Loading checkpoint from {repo_id}/{branch_name}...")
        local_dir = snapshot_download(repo_id=repo_id, revision=branch_name, repo_type="model")

        data: Dict[str, Any] = {"local_dir": local_dir}

        # Params
        pth = os.path.join(local_dir, "model_params.flax")
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                data["params"] = from_bytes(None, f.read())

        # Opt states
        if load_optimizer_states:
            ep = os.path.join(local_dir, "opt_state_embed.flax")
            if os.path.exists(ep):
                with open(ep, "rb") as f:
                    data["opt_state_embed"] = from_bytes(None, f.read())
            op = os.path.join(local_dir, "opt_state_other.flax")
            if os.path.exists(op):
                with open(op, "rb") as f:
                    data["opt_state_other"] = from_bytes(None, f.read())

        # Metadata & config
        mp = os.path.join(local_dir, "training_metadata.json")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as f:
                data["metadata"] = json.load(f)
        cp = os.path.join(local_dir, "config.json")
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                data["config"] = json.load(f)

        logger.info(f"✅ Checkpoint loaded from {repo_id}/{branch_name}")
        return data
    except Exception as e:
        logger.error(f"Failed to load checkpoint from hub: {str(e)}")
        return None

# ----------------------------
# Milestones & cleanup
# ----------------------------
def get_word_milestone_name(words_processed: int) -> str:
    if words_processed >= 1_000_000:
        return f"checkpoint_{words_processed // 1_000_000}M_words"
    if words_processed >= 1_000:
        return f"checkpoint_{words_processed // 1_000}K_words"
    return f"checkpoint_{words_processed}_words"

def should_save_checkpoint(
    words_processed: int,
    last_checkpoint_words: int,
    checkpoint_interval_words: int = 1_000_000,
) -> bool:
    return (words_processed - last_checkpoint_words) >= checkpoint_interval_words

def cleanup_old_checkpoints(checkpoint_dir: str, keep_n: int = 5):
    try:
        if not os.path.exists(checkpoint_dir):
            return
        files = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith("checkpoint_") or f.startswith("training_state_"):
                p = os.path.join(checkpoint_dir, f)
                try:
                    mtime = os.path.getmtime(p)
                    files.append((p, mtime))
                except Exception:
                    continue
        files.sort(key=lambda x: x[1], reverse=True)
        for p, _ in files[keep_n:]:
            try:
                os.remove(p)
                logger.info(f"Removed old checkpoint: {p}")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {str(e)}")

# ----------------------------
# Final model export
# ----------------------------
def save_final_model(
    params: Dict,
    config: Any,
    tokenizer,
    save_dir: str,
    create_hf_compatible: bool = True,
):
    """
    Save final model (Flax params + config + tokenizer + README).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Params
    with open(os.path.join(save_dir, "model.flax"), "wb") as f:
        f.write(to_bytes(params))

    # Config
    write_config(config, save_dir)

    # Tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    # Model card
    model_card = f"""---
language: en
tags: [structformer, hyperbolic, poincare, language-modeling, babylm]
license: mit
---

# StructFormer + Poincaré Embeddings

A model combining StructFormer’s structure induction with Poincaré (hyperbolic) token embeddings.

## Training
- Framework: JAX/Flax
- Optimization: Dual (AdamW for model, Riemannian for embeddings)
- Losses: Cross-entropy + Hyperbolic regularizers

## Loading
```python
from flax.serialization import from_bytes
with open("model.flax","rb") as f:
    params = from_bytes(None, f.read())
# Instantiate your model class with the same config used for training.
"""
    with open(os.path.join(save_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info(f"✅ Final model saved to {save_dir}")
