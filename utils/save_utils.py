import os
import json
import jax
import numpy as np
import subprocess
from tempfile import TemporaryDirectory
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file as save_safetensors
from huggingface_hub import HfApi, create_repo


# -------------------------------------------------
#  Utility: Flatten and sanitize params for saving
# -------------------------------------------------
def sanitize(params):
    """Filter object-dtype or invalid leaves from a Flax PyTree for saving."""
    flat = flatten_dict(jax.tree_util.tree_map(lambda x: x, params), sep="/")
    safe = {}
    for k, v in flat.items():
        if isinstance(v, (np.ndarray, jax.Array)) and getattr(v, "dtype", None) != object:
            key = "/".join(k) if isinstance(k, tuple) else k
            safe[key] = np.array(v)
        else:
            print(f"⚠️ Skipping {k} (dtype={getattr(v, 'dtype', type(v))})")
    return safe


# -------------------------------------------------
#  Write config.json with correct auto_map reference
# -------------------------------------------------
def write_config(config_dict: dict, destination: str, model_file: str = "models/structformer_poincare"):
    """
    Write config.json including auto_map for Flax AutoModel loading from HF Hub.
    model_file should be the Python module path (without .py) to your model class.
    """
    export_config = dict(config_dict)
    export_config.update({
        "model_type": "structformer",
        "auto_map": {
            "AutoConfig": "structformer_config.StructformerConfig",
            "FlaxAutoModel": f"{model_file}.StructformerPoincare",
            "FlaxAutoModelForMaskedLM": f"{model_file}.StructformerPoincare"
        }
    })
    config_path = os.path.join(destination, "config.json")
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    print(f"✅ Wrote config.json to {config_path}")


# -------------------------------------------------
#  Push model to HF Hub main branch
# -------------------------------------------------
def export_model_to_huggingface(
    params,
    config,
    repo_id,
    commit_message="Exported final model",
    include_modeling_files=None,
):
    """
    Push model weights, config, and source files to HF Hub's main branch.
    include_modeling_files: list of .py files needed for model import (e.g., 
        ['models/structformer_poincare.py', 'models/hyperbolic_layers.py'])
    """
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()
    with TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "flax_model.safetensors")
        config_path = os.path.join(tmpdir, "config.json")

        # Save sanitized parameters
        safe_params = sanitize(params)
        save_safetensors(safe_params, model_path)
        print(f"✅ Saved flax_model.safetensors")

        # Write config.json
        write_config(config, tmpdir, model_file="models/structformer_poincare")

        # Upload model weights & config to HF Hub
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=model_path,
            path_in_repo="flax_model.safetensors",
            repo_type="model",
            commit_message=commit_message
        )
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_type="model",
            commit_message=commit_message
        )

        # Upload any additional necessary source files for model import
        if include_modeling_files:
            for fname in include_modeling_files:
                if os.path.exists(fname):
                    api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=fname,
                        path_in_repo=os.path.basename(fname),
                        repo_type="model",
                        commit_message=f"Include source: {fname}"
                    )
                    print(f"📄 Uploaded ⤴ {fname}")
                else:
                    print(f"⚠️ File not found: {fname}")

    print(f"🎉 Model + config pushed to: https://huggingface.co/{repo_id}")


# -------------------------------------------------
#  Save checkpoint to specific HF Hub branch
# -------------------------------------------------
def save_checkpoint_branch(
    params,
    config,
    branch_name,
    repo_id,
    include_modeling_files=None,
    model_file: str = "models/structformer_poincare",
):
    """
    Save a checkpoint to a dedicated HF Hub branch (useful for mid-training checkpoints).
    """
    with TemporaryDirectory() as tmp:
        model_file_path = os.path.join(tmp, "flax_model.safetensors")

        # Save params
        safe_params = sanitize(params)
        save_safetensors(safe_params, model_file_path)
        # Write config
        write_config(config, tmp, model_file=model_file)

        # Copy modeling files into temporary dir
        if include_modeling_files:
            for f in include_modeling_files:
                if os.path.exists(f):
                    subprocess.run(["cp", f, os.path.join(tmp, os.path.basename(f))], check=True)
                else:
                    print(f"⚠️ File not found: {f}")

        # Init git repo, commit, and push to branch
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            subprocess.run(["git", "config", "user.name", "HF Trainer"], check=True)
            subprocess.run(["git", "config", "user.email", "trainer@hf.local"], check=True)
            subprocess.run(["git", "lfs", "install"], check=True)
            subprocess.run(["git", "lfs", "track", "*.safetensors"], check=True)
            subprocess.run(["git", "add", ".gitattributes"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Add {branch_name} checkpoint"], check=True)
            subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/{repo_id}"], check=True)
            subprocess.run(["git", "push", "--force", "origin", branch_name], check=True)
        finally:
            os.chdir(cwd)

        print(f"✅ Checkpoint pushed to branch: https://huggingface.co/{repo_id}/tree/{branch_name}")


# -------------------------------------------------
#  Extra utilities for batching & milestones
# -------------------------------------------------
def batch_data(ids, masks, batch_size):
    """Yield batches from full dataset arrays."""
    for i in range(0, len(ids), batch_size):
        yield {
            "input_ids": ids[i:i+batch_size],
            "attention_mask": masks[i:i+batch_size],
        }


def make_milestones(max_words=100_000_000):
    """Create milestone steps for logging/training monitors."""
    return (
        list(range(1_000_000, 10_000_001, 1_000_000)) +
        list(range(20_000_000, max_words + 1, 10_000_000))
    )
