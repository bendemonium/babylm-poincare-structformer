import argparse
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from models.structformer_poincare import StructformerPoincare
from utils.train_utils import (
    create_train_states,
    train_step,
    eval_epoch
)
from utils.logging_utils import MetricLogger
from utils.save_utils import (
    make_milestones,
    save_checkpoint_branch,
    export_model_to_huggingface,
)
from utils.retrieve_utils import load_dataset_dynamic


def count_tokens_in_batch(batch):
    """Count actual tokens in batch (excluding padding)."""
    return int(jnp.sum(batch["attention_mask"]))


def load_config_from_yaml(config_path):
    """Load YAML config and convert to dot-accessible object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class DictAsAttr:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, DictAsAttr(v))
                else:
                    setattr(self, k, v)
        
        def __getitem__(self, key):
            return getattr(self, key)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    return DictAsAttr(config_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Run only a few batches for testing")
    args = parser.parse_args()

    # Load config
    config = load_config_from_yaml(args.config)
    
    # Environment variables
    HF_REPO_ID = os.environ.get("HF_REPO_ID", config.checkpointing.output_repo_id)
    
    print(f"🎯 Training with word-based milestones up to {config.training.max_words//1_000_000}M words")

    # Load datasets
    print("📥 Loading datasets...")
    train_data = load_dataset_dynamic(
        config.data.train_tokenized_repo,
        method="pickle",
        pickle_filename=config.data.train_tokenized_file
    )
    val_data = load_dataset_dynamic(
        config.data.val_tokenized_repo,
        method="pickle",
        pickle_filename=config.data.val_tokenized_file
    )

    print(f"✅ Loaded training dataset with {len(train_data['input_ids']):,} examples")
    print(f"✅ Loaded validation dataset with {len(val_data['input_ids']):,} examples")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(config.data.vocab_name)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    
    # Initialize model and dual states
    rng = jax.random.PRNGKey(config.system.seed)
    model = StructformerPoincare(
        vocab_size=vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        max_length=config.model.max_length,
        c=config.model.c
    )

    state_embed, state_other = create_train_states(
        rng, model,
        vocab_size=vocab_size,
        seq_len=config.model.max_length,
        lr_ce=config.training.lr_ce,
        lr_riem=config.training.lr_riem
    )

    # Logging setup
    logger = MetricLogger(
        log_dir=config.logging.log_dir,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_runname=config.logging.wandb_run_name
    )

    # Word-based training setup
    total_words_processed = 0
    milestones = make_milestones(max_words=config.training.max_words)
    next_milestone_idx = 0
    next_eval_words = config.training.eval_interval_words
    next_log_words = config.logging.log_interval_words
    global_step = 0
    
    print(f"🎯 Training milestones: {[f'{m//1_000_000}M' for m in milestones[:5]]}...")
    
    # Training data setup
    total_samples = len(train_data["input_ids"])
    indices = np.random.permutation(total_samples)
    batch_idx = 0

    # Dry run limit
    max_words_for_run = 1_000_000 if args.dry_run else config.training.max_words
    
    # Main training loop - word-based
    print(f"\n🚀 Starting word-based training...")
    with tqdm(total=max_words_for_run, unit="words", desc="Training Progress") as pbar:
        while total_words_processed < max_words_for_run:
            # Create batch indices
            start_idx = (batch_idx * config.training.batch_size) % total_samples
            end_idx = min(start_idx + config.training.batch_size, total_samples)
            
            # Handle wrap-around and reshuffle
            if end_idx - start_idx < config.training.batch_size:
                indices = np.random.permutation(total_samples)
                start_idx = 0
                end_idx = config.training.batch_size
                print("🔄 Reshuffled training data")

            idx = indices[start_idx:end_idx]
            
            # Create batch
            batch = {
                "input_ids": jnp.array(np.array(train_data["input_ids"])[idx], dtype=jnp.int32),
                "attention_mask": jnp.array(np.array(train_data["attention_mask"])[idx], dtype=jnp.bool_)
            }

            # Training step with dual states
            state_embed, state_other, train_metrics = train_step(
                state_embed, state_other, batch,
                c=config.model.c,
                lambda_poincare=config.training.lambda_poincare,
                model=model
            )

            # Count words processed in this batch
            batch_words = count_tokens_in_batch(batch)
            total_words_processed += batch_words
            global_step += 1
            batch_idx += 1
            
            # Update progress bar
            pbar.update(batch_words)
            pbar.set_postfix({
                'CE Loss': f"{float(train_metrics['loss_ce']):.4f}",
                'Poincare Loss': f"{float(train_metrics['loss_poincare']):.4f}",
                'Total Loss': f"{float(train_metrics['loss_total']):.4f}"
            })

            # Logging at intervals
            if total_words_processed >= next_log_words:
                logger.log_metrics(train_metrics, step=total_words_processed, prefix="train")
                next_log_words += config.logging.log_interval_words

            # Validation at intervals
            if total_words_processed >= next_eval_words:
                print(f"\n🔍 Running validation at {total_words_processed//1_000_000}M words...")
                eval_metrics = eval_epoch(
                    state_embed, state_other, val_data,
                    config.training.eval_batch_size,
                    config.model.c,
                    config.training.lambda_poincare,
                    model,
                    dataset_type="pickle",
                    max_batches=100
                )
                logger.log_metrics(eval_metrics, step=total_words_processed, prefix="val")
                print(f"📊 Validation: CE Loss: {eval_metrics['loss_ce']:.4f}, Poincaré Loss: {eval_metrics['loss_poincare']:.4f}")
                next_eval_words += config.training.eval_interval_words

            # Checkpoint at milestones
            if (next_milestone_idx < len(milestones) and 
                total_words_processed >= milestones[next_milestone_idx]):
                
                milestone_words = milestones[next_milestone_idx]
                branch_name = f"{config.checkpointing.branch_prefix}_{milestone_words//1_000_000}M_words"
                
                print(f"\n💾 Saving checkpoint at {milestone_words//1_000_000}M words...")
                
                # Merge parameters for saving
                merged_params = {**state_embed.params, **state_other.params}
                
                save_checkpoint_branch(
                    params={"params": merged_params},
                    config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
                    branch_name=branch_name,
                    repo_id=HF_REPO_ID,
                    include_modeling_files=config.checkpointing.include_modeling_files,
                    model_file=config.checkpointing.model_file
                )
                next_milestone_idx += 1

            # Progress logging every 1000 steps
            if global_step % 1000 == 0:
                progress_pct = (total_words_processed / max_words_for_run) * 100
                print(f"📊 Step {global_step}: {total_words_processed//1_000_000}M/{max_words_for_run//1_000_000}M words ({progress_pct:.1f}%)")

    # Final export if requested
    if os.getenv("HF_MODEL_EXPORT", "no").lower() == "yes" and not args.dry_run:
        print("\n📤 Exporting final model to Hugging Face Hub...")
        merged_params = {**state_embed.params, **state_other.params}
        export_model_to_huggingface(
            params={"params": merged_params},
            config=config.__dict__ if hasattr(config, '__dict__') else vars(config),
            repo_id=HF_REPO_ID,
            commit_message=f"Final model after {total_words_processed:,} words",
            include_modeling_files=config.checkpointing.include_modeling_files
        )

    logger.close()
    print(f"✅ Training completed! Processed {total_words_processed//1_000_000}M words")


if __name__ == "__main__":
    main()
