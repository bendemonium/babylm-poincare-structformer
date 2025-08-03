# utils/export_hf.py

import os
import json
import jax
import numpy as np
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file as save_safetensors
from huggingface_hub import HfApi, create_repo


def sanitize(params):
    """Remove object-dtype or invalid leaves from a Flax PyTree."""
    flat = flatten_dict(jax.tree_util.tree_map(lambda x: x, params), sep="/")
    safe = {}

    for k, v in flat.items():
        if isinstance(v, (np.ndarray, jax.Array)) and getattr(v, "dtype", None) != object:
            key = "/".join(k) if isinstance(k, tuple) else k
            safe[key] = np.array(v)
        else:
            print(f"⚠️ Skipping {k} (dtype={getattr(v, 'dtype', type(v))})")

    return safe


def write_config(config: dict, destination: str):
    export_config = {
        "auto_map": {
            "AutoConfig": "model_configuration.ModelConfig",
            "AutoModel": "modeling.MyModel",
            "AutoModelForCausalLM": "modeling.MyModelForCausalLM"
        },
        "attention_dropout_p": 0.1,
        "attention_bias": True,
        "attention_layernorm_learned": True,
        "embedding_dropout_prob": 0.1,
        "embedding_layernorm_learned": True,
        "hidden_size": config.get("hidden_dim", 384),
        "intermediate_size": config.get("intermediate_size", 1024),
        "layernorm_eps": 1e-5,
        "lm_head_layernorm_learned": True,
        "max_sequence_length": config.get("seq_length", 512),
        "mlp_bias": True,
        "mlp_layernorm_learned": True,
        "num_attention_heads": config.get("num_heads", 6),
        "num_layers": config.get("num_layers", 6),
        "rope_theta": 10000,
        "vocab_size": config.get("vocab_size", 6144)
    }

    path = os.path.join(destination, "config.json")
    with open(path, "w") as f:
        json.dump(export_config, f, indent=2)


def export_model_to_huggingface(params, config, repo_id, commit_message="Exported final model"):
    from tempfile import TemporaryDirectory

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

    print(f"✅ Uploaded final model + config to HF at: {repo_id}")
