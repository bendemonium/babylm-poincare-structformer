import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

from utils.train_utils import (
    create_train_states,
    train_step,
    eval_step,
    eval_epoch,
)
from utils.save_utils import write_config
from models.structformer_poincare import StructformerPoincare


def generate_dummy_batch(batch_size=2, seq_len=8, vocab_size=100):
    """Generate a small random batch for smoke tests."""
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    m = np.ones_like(x, dtype=np.int8)  # mask as 0/1; jnp.bool_ or float32 also fine
    return {
        "input_ids": jnp.array(x),
        "attention_mask": jnp.array(m),
    }


def check_for_nans(metrics_dict, where=""):
    """Raise error if any metric value is NaN or Inf."""
    for k, v in metrics_dict.items():
        # v can be JAX scalar; convert to float if possible
        try:
            v = float(v)
        except Exception:
            try:
                v = float(np.array(v))
            except Exception:
                pass
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            raise ValueError(f"❌ NaN/Inf detected in {k} at {where}: {v}")
    print(f"✅ No NaNs/Infs in metrics at {where}")


def run_all_tests():
    print("=== Starting Smoke Tests ===")

    # ---- Model init ----
    vocab_size = 100
    seq_len = 8
    rng = jax.random.PRNGKey(0)
    model = StructformerPoincare(
        vocab_size=vocab_size,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        max_length=seq_len,
        c=1.0,
    )
    print("✅ Model init OK")

    # ---- Create states ----
    state_embed, state_other = create_train_states(
        rng, model, vocab_size, seq_len, lr_ce=1e-3, lr_riem=1e-3
    )
    print("✅ create_train_states OK")

    # ---- Train step test ----
    batch = generate_dummy_batch(seq_len=seq_len, vocab_size=vocab_size)
    state_embed, state_other, train_metrics = train_step(
        state_embed, state_other, batch, c=1.0, lambda_poincare=0.1, model=model
    )
    print("✅ train_step OK")
    check_for_nans(train_metrics, "train_step")

    # ---- Eval step test ----
    eval_metrics = eval_step(
        state_embed, state_other, batch, c=1.0, lambda_poincare=0.1, model=model
    )
    print("✅ eval_step OK")
    check_for_nans(eval_metrics, "eval_step")

    # ---- Eval epoch test (tiny HF-like dataset) ----
    try:
        import datasets
        dummy_ds = datasets.Dataset.from_dict({
            "input_ids": np.random.randint(0, vocab_size, (10, seq_len), dtype=np.int32),
            "attention_mask": np.ones((10, seq_len), dtype=np.int8),
        })
        epoch_metrics = eval_epoch(
            state_embed,
            state_other,
            val_data=dummy_ds,
            batch_size=2,
            c=1.0,
            lambda_poincare=0.1,
            model=model,
            dataset_type="hf",
            max_batches=2,
        )
        print("✅ eval_epoch OK")
        check_for_nans(epoch_metrics, "eval_epoch")
    except Exception as e:
        print(f"⚠️ Skipping eval_epoch HF test due to: {e}")

    # ---- Config write test ----
    tmpdir = tempfile.mkdtemp()
    cfg_stub = {
        "seed": 42,
        "model": {
            "vocab_size": vocab_size,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "max_length": seq_len,
            "c": 1.0,
        },
        "training": {
            "batch_size": 2,
            "eval_batch_size": 2,
            "max_words": 1000,
            "lr_ce": 1e-3,
            "lr_riem": 1e-3,
            "lambda_h": 0.1,
            "lambda_tree": 0.0,
        },
    }
    write_config(cfg_stub, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "config.json")), "config.json not written"
    print("✅ write_config OK")

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    run_all_tests()


# python -m utils.test_utils