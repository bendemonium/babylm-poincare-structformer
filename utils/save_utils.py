"""
Save utilities for StructFormer + Poincaré training.
Handles:
  • Local checkpoints
  • HF Hub branch uploads (fully Transformers/Flax-compatible)
  • Final model export

On each branch upload we include:
  - config.json (HF-style, with architectures)
  - flax_model.safetensors        (primary Flax weights format)
  - flax_model.msgpack            (legacy Flax weights, for broad compatibility)
  - model_params.flax             (legacy file name our older tools expect)
  - opt_state_embed.flax / opt_state_other.flax (optional optimizer states)
  - training_metadata.json
  - README.md
  - (optional) modeling source files (for trust_remote_code=True)

Notes:
- Tokenizer artifacts are NOT uploaded by default since we use the stock "gpt2" tokenizer.
- If you ever change vocab/merges/special tokens, upload tokenizer files alongside the branch.
"""

import os
import json
import shutil
import tempfile
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import jax
from flax.training import checkpoints
from flax.serialization import to_bytes, from_bytes

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    snapshot_download,
)
from huggingface_hub.errors import HfHubHTTPError
from safetensors.flax import save_file as save_safetensors

logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _is_jax_array(x: Any) -> bool:
    # Works across recent JAX versions (Array) and older DeviceArray naming.
    return isinstance(x, jax.Array) or x.__class__.__name__ in {"Array", "DeviceArray"}

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable forms.
    Summarizes large arrays to avoid huge JSON files.
    """
    try:
        if hasattr(obj, "__dict__") and not isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
        if hasattr(obj, "_asdict"):
            return {k: make_json_serializable(v) for k, v in obj._asdict().items()}
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        if _is_jax_array(obj):
            try:
                size = int(obj.size)
                if size <= 100:
                    return obj.tolist()
                return f"jax.Array(shape={tuple(obj.shape)}, dtype={obj.dtype})"
            except Exception:
                return "jax.Array"
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)
    except Exception:
        return str(obj)

def sanitize(obj: Any) -> Any:
    return make_json_serializable(obj)

# =============================================================================
# HF config.json (Transformers-compatible)
# =============================================================================

def build_hf_config_dict(cfg: Any) -> dict:
    """
    Build a Transformers-compatible config.json.
    Expects a SimpleNamespace-like object with attributes used below.
    """
    return {
        "model_type": "structformer_poincare",
        "architectures": ["FlaxStructformerPoincareForMaskedLM"],  # used by AutoModel to pick class
        "hidden_dim": int(getattr(cfg, "hidden_dim", 256)),
        "num_layers": int(getattr(cfg, "num_layers", 8)),
        "num_heads": int(getattr(cfg, "num_heads", 8)),
        "max_length": int(getattr(cfg, "max_length", 128)),
        "dropout_rate": float(getattr(cfg, "dropout_rate", 0.1)),
        "c": float(getattr(cfg, "c", 1.0)),
        "attention_input": getattr(cfg, "attention_input", "tangent"),
        "pad_token_id": int(getattr(cfg, "pad_id", 50256)),  # GPT-2 often uses eos as pad
        # Optional / informational:
        "task_specific_params": {"masked-language-modeling": {}},
        "torch_dtype": "float32",
    }

def write_hf_config_json(cfg: Any, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    cfg_dict = build_hf_config_dict(cfg)
    path = os.path.join(save_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    logger.info("✅ HF config.json written to %s", path)
    return path

# =============================================================================
# Local checkpoints (Flax)
# =============================================================================

def save_flax_checkpoint(
    params: Dict,
    opt_state_embed: Any,
    opt_state_other: Any,
    step: int,
    save_dir: str,
    prefix: str = "checkpoint",
) -> None:
    """
    Save a local Flax checkpoint with params + optimizer states.
    """
    os.makedirs(save_dir, exist_ok=True)
    target = {
        "params": params,
        "opt_state_embed": opt_state_embed,
        "opt_state_other": opt_state_other,
        "step": int(step),
        "framework": "jax_flax",
    }
    checkpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=target,
        step=step,
        prefix=prefix,
        keep=5,
        overwrite=True,
    )
    logger.info("✅ Flax checkpoint saved: %s/%s_%s", save_dir, prefix, step)

def load_flax_checkpoint(
    ckpt_dir: str,
    step: Optional[int] = None,
    prefix: str = "checkpoint",
) -> Optional[Dict]:
    """
    Load a local Flax checkpoint (latest if step is None).
    """
    if not os.path.exists(ckpt_dir):
        logger.warning("Checkpoint directory not found: %s", ckpt_dir)
        return None
    data = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None,
        step=step,
        prefix=prefix,
    )
    if data is None:
        logger.warning("No checkpoint found in %s", ckpt_dir)
        return None
    logger.info("✅ Loaded checkpoint from %s, step %s", ckpt_dir, data.get("step", "unknown"))
    return data

# =============================================================================
# HF Hub helpers
# =============================================================================

def _ensure_hf_repo(repo_id: str, private: bool = True, repo_type: str = "model") -> None:
    api = HfApi()
    try:
        _ = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        logger.info("HF repo %s exists", repo_id)
        return
    except Exception:
        pass
    create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
    logger.info("✅ Created HF repo: %s", repo_id)

def _ensure_hf_branch(repo_id: str, branch: str, repo_type: str = "model") -> None:
    api = HfApi()
    try:
        api.create_branch(repo_id=repo_id, branch=branch, repo_type=repo_type)
        logger.info("✅ Created branch %s for %s", branch, repo_id)
    except HfHubHTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 409:
            # Already exists
            logger.info("Branch %s already exists for %s", branch, repo_id)
        else:
            raise

def copy_modeling_files(source_files: List[str], target_dir: str) -> None:
    """
    Copy selected modeling files into upload folder for trust_remote_code=True.
    """
    for src in source_files or []:
        if not os.path.exists(src):
            logger.warning("Modeling file not found: %s", src)
            continue
        dst = os.path.join(target_dir, os.path.normpath(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Copied %s → %s", src, dst)

# =============================================================================
# Branch upload (HF-compatible)
# =============================================================================

def save_checkpoint_branch(
    params: Dict,
    config: Any,
    branch_name: str,
    repo_id: str,
    include_modeling_files: Optional[List[str]] = None,
    model_file: Optional[str] = None,
    opt_state_embed: Any = None,
    opt_state_other: Any = None,
    metrics: Optional[Dict] = None,
    step: int = 0,
    words_processed: int = 0,
) -> None:
    """
    Package and upload a checkpoint to a specific Hub branch.

    Files written:
      - config.json
      - flax_model.safetensors
      - flax_model.msgpack
      - model_params.flax
      - opt_state_embed.flax (optional)
      - opt_state_other.flax (optional)
      - training_metadata.json
      - README.md
      - (optional) modeling files
    """
    logger.info("Saving checkpoint to branch %s...", branch_name)
    _ensure_hf_repo(repo_id, private=True, repo_type="model")
    _ensure_hf_branch(repo_id, branch_name, repo_type="model")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1) HF config.json (Transformers reads this)
        write_hf_config_json(config, tmp_dir)

        # 2) Flax weights — primary: safetensors; also include msgpack + legacy file name.
        #    a) flax_model.safetensors
        safetensors_path = os.path.join(tmp_dir, "flax_model.safetensors")
        save_safetensors(params, safetensors_path)

        #    b) flax_model.msgpack (Transformers still supports this format broadly)
        flax_msgpack_path = os.path.join(tmp_dir, "flax_model.msgpack")
        with open(flax_msgpack_path, "wb") as f:
            f.write(to_bytes(params))

        #    c) model_params.flax (legacy name some of our internal tools used)
        legacy_params_path = os.path.join(tmp_dir, "model_params.flax")
        with open(legacy_params_path, "wb") as f:
            f.write(to_bytes(params))

        # 3) Optimizer states (optional)
        if opt_state_embed is not None:
            with open(os.path.join(tmp_dir, "opt_state_embed.flax"), "wb") as f:
                f.write(to_bytes(opt_state_embed))
        if opt_state_other is not None:
            with open(os.path.join(tmp_dir, "opt_state_other.flax"), "wb") as f:
                f.write(to_bytes(opt_state_other))

        # 4) Training metadata (useful for dashboards / resume)
        metadata = {
            "step": int(step),
            "words_processed": int(words_processed),
            "branch_name": branch_name,
            "framework": "jax_flax",
            "timestamp": _now_iso(),
        }
        if metrics is not None:
            metadata["metrics"] = make_json_serializable(metrics)

        with open(os.path.join(tmp_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 5) Optional modeling files (for trust_remote_code=True)
        if include_modeling_files:
            copy_modeling_files(include_modeling_files, tmp_dir)

        # 6) README (model card-lite)
        readme = f"""# StructFormer + Poincaré — Checkpoint

Checkpoint saved during training.

**Repo**: `{repo_id}`  
**Branch**: `{branch_name}`  
**Step**: {step:,}  
**Words processed**: {words_processed:,}  
**Timestamp**: {metadata['timestamp']}

## Load (Flax)

```python
from transformers import AutoTokenizer, FlaxAutoModelForMaskedLM
import jax.numpy as jnp

repo = "{repo_id}"
branch = "{branch_name}"

# Using stock GPT-2 tokenizer (unchanged)
tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

model = FlaxAutoModelForMaskedLM.from_pretrained(
    repo, revision=branch, trust_remote_code=True, dtype=jnp.float32
)
```
## Files
- `config.json`                (Transformers config)
- `flax_model.safetensors`     (Flax weights, primary)
- `flax_model.msgpack`         (Flax weights, legacy msgpack)
- `model_params.flax`          (legacy filename kept for internal tools)
- `opt_state_embed.flax`       (optional)
- `opt_state_other.flax`       (optional)
- `training_metadata.json`
- modeling source files (if included)
"""
        with open(os.path.join(tmp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

        # 7) Upload to the branch
        logger.info("[HF upload] repo=%s branch=%s", repo_id, branch_name)
        upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=branch_name,  # branch already ensured
            commit_message=f"Checkpoint at {words_processed:,} words (step {step:,})",
            create_pr=False,
        )
        logger.info("✅ Checkpoint saved to %s/%s", repo_id, branch_name)

# =============================================================================
# Resume from Hub
# =============================================================================

def load_checkpoint_from_hub(
    repo_id: str,
    branch_name: str = "main",
    load_optimizer_states: bool = True,
) -> Optional[Dict]:
    """
    Download a branch snapshot and read params/opt states/metadata/config.
    Prefers `flax_model.msgpack` / `model_params.flax` for internal resume.
    (Transformers will handle safetensors automatically on its side.)
    """
    try:
        logger.info("Loading checkpoint from %s/%s...", repo_id, branch_name)
        local_dir = snapshot_download(repo_id=repo_id, revision=branch_name, repo_type="model")

        data: Dict[str, Any] = {"local_dir": local_dir}

        # Prefer msgpack for internal resume:
        p_msgpack = os.path.join(local_dir, "flax_model.msgpack")
        p_legacy  = os.path.join(local_dir, "model_params.flax")

        params_bytes = None
        if os.path.exists(p_msgpack):
            with open(p_msgpack, "rb") as f:
                params_bytes = f.read()
        elif os.path.exists(p_legacy):
            with open(p_legacy, "rb") as f:
                params_bytes = f.read()

        if params_bytes is not None:
            data["params"] = from_bytes(None, params_bytes)
        else:
            logger.warning("No msgpack params found in %s (will rely on safetensors via Transformers).", local_dir)

        # Optimizer states (optional)
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

        logger.info("✅ Checkpoint loaded from %s/%s", repo_id, branch_name)
        return data
    except Exception as e:
        logger.error("Failed to load checkpoint from hub: %s", str(e))
        return None

# =============================================================================
# Milestones & cleanup
# =============================================================================

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

def cleanup_old_checkpoints(checkpoint_dir: str, keep_n: int = 5) -> None:
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
                logger.info("Removed old checkpoint: %s", p)
            except Exception:
                pass
    except Exception as e:
        logger.error("Failed to cleanup checkpoints: %s", str(e))

# =============================================================================
# Final model export (optional local artifact)
# =============================================================================

def save_final_model(
    params: Dict,
    config: Any,
    tokenizer,  # optional; we usually rely on "gpt2" upstream
    save_dir: str,
) -> None:
    """
    Save a local, HF-compatible Flax export (not uploaded automatically).
    Includes:
      - config.json
      - flax_model.safetensors
      - flax_model.msgpack
      - (optional) tokenizer files if provided
      - README.md
    """
    os.makedirs(save_dir, exist_ok=True)

    # Config
    write_hf_config_json(config, save_dir)

    # Weights (both formats)
    save_safetensors(params, os.path.join(save_dir, "flax_model.safetensors"))
    with open(os.path.join(save_dir, "flax_model.msgpack"), "wb") as f:
        f.write(to_bytes(params))

    # Tokenizer (optional – not required if using stock GPT-2 unchanged)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    # Model card-lite
    model_card = f"""# StructFormer + Poincaré (Flax)

Local export created at { _now_iso() }.

Files:
- `config.json`
- `flax_model.safetensors`
- `flax_model.msgpack`
- tokenizer files (if provided)

## Load (Flax)
```python
from transformers import AutoTokenizer, FlaxAutoModelForMaskedLM
import jax.numpy as jnp

tok = AutoTokenizer.from_pretrained("{'gpt2' if tokenizer is None else save_dir}", use_fast=True)
model = FlaxAutoModelForMaskedLM.from_pretrained("{save_dir}", trust_remote_code=True, dtype=jnp.float32)
```
"""
    with open(os.path.join(save_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)

    logger.info("✅ Final model saved to %s", save_dir)
