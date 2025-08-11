# test_save_checkpoint_branch.py
import jax.numpy as jnp
from flax.core import freeze
from flax import serialization
from utils.save_utils import save_checkpoint_branch

# -----------------------------
# Dummy data for checkpoint
# -----------------------------
# Fake model params (e.g., small embedding table)
params = freeze({
    "embed_table": jnp.ones((5, 5)),  # small matrix for test
    "layer": {
        "weights": jnp.zeros((2, 2)),
        "bias": jnp.array([1.0, -1.0])
    }
})

# Fake optimizer states
opt_state_embed = {"step": 1, "state": "embed_opt"}
opt_state_other = {"step": 1, "state": "other_opt"}

# Minimal config object
class DummyConfig:
    c = 1.0
    checkpointing = {"output_repo_id": "YOUR_HF_USERNAME/test-checkpoint-repo"}
    logging = {}

config = DummyConfig()

# -----------------------------
# Run save
# -----------------------------
save_checkpoint_branch(
    params=params,
    config=config,
    branch_name="test_dummy_checkpoint",
    repo_id="bendemonium/babylm25-poincare-structformer",  # <-- change to your repo
    include_modeling_files=None,
    model_file=None,
    opt_state_embed=opt_state_embed,
    opt_state_other=opt_state_other,
    metrics={"loss": 0.123},
    step=1,
    words_processed=100
)

print("✅ Dummy checkpoint save attempted.")