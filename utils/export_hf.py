import os
import json
import jax
import numpy as np
from tempfile import TemporaryDirectory
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file as save_safetensors
from huggingface_hub import HfApi, create_repo


def sanitize(params):
    """
    Filter object-dtype or invalid leaves from a Flax PyTree for saving.
    """
    flat = flatten_dict(jax.tree_util.tree_map(lambda x: x, params), sep="/")
    safe = {}

    for k, v in flat.items():
        if isinstance(v, (np.ndarray, jax.Array)) and getattr(v, "dtype", None) != object:
            key = "/".join(k) if isinstance(k, tuple) else k
            safe[key] = np.array(v)
        else:
            print(f"⚠️ Skipping {k} (dtype={getattr(v, 'dtype', type(v))})")

    return safe


def write_config(config: dict, destination: str, model_file: str = "modeling.FlaxStructformerModel"):
    """
    Export config.json with auto_map entries for custom Flax model.
    """
    export_config = {
        "model_type": "structformer",
        "auto_map": {
            "AutoConfig": "structformer_config.StructformerConfig",
            "FlaxAutoModel": model_file,
            "FlaxAutoModelForMaskedLM": model_file
        },
        "hidden_dim": config.get("hidden_dim", 512),
        "num_heads": config.get("num_heads", 8),
        "num_layers": config.get("num_layers", 6),
        "max_length": config.get("seq_length", 128),
        "vocab_size": config.get("vocab_size", 50257),
        "c": config.get("c", 1.0)
    }

    path = os.path.join(destination, "config.json")
    with open(path, "w") as f:
        json.dump(export_config, f, indent=2)


def export_model_to_huggingface(params, config, repo_id, *, commit_message="Exported final model"):
    """
    Sanitize + export Flax model checkpoint to Hugging Face Hub branch or main repo.
    """
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()

    with TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "flax_model.safetensors")
        config_path = os.path.join(tmpdir, "config.json")

        safe_params = sanitize(params)
        save_safetensors(safe_params, model_path)
        write_config(config, tmpdir)

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

    print(f"✅ Uploaded flax_model.safetensors + config.json → {repo_id}")

