import os
import json
import time

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class MetricLogger:
    """
    Simple, flexible logger for training & validation metrics.
    Supports:
      - Console logs
      - TensorBoard
      - Weights & Biases
    """

    def __init__(self, log_dir=None, use_wandb=False, wandb_project=None, wandb_runname=None):
        self.start_time = time.time()
        self.tb_writer = None
        self.wandb_active = False

        # --- TensorBoard ---
        if log_dir and _TB_AVAILABLE:
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"📝 TensorBoard logging to {log_dir}")

        # --- Weights & Biases ---
        if use_wandb and _WANDB_AVAILABLE:
            wandb.init(project=wandb_project or "project", name=wandb_runname)
            self.wandb_active = True
            print(f"📡 Weights & Biases logging to project: {wandb_project}")

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """
        Log a dictionary of metrics.
        Args:
            metrics: dict of metrics {'loss_total': ..., 'loss_ce': ...}
            step: global step or epoch number
            prefix: optional string to prefix all metric names (e.g., "train" or "val")
        """
        elapsed = time.time() - self.start_time
        prefix_str = f"{prefix}_" if prefix else ""

        # --- Console ---
        metric_str = " | ".join([f"{prefix_str}{k}: {float(v):.4f}" for k, v in metrics.items()])
        print(f"[step {step}] {metric_str} | elapsed: {elapsed:.1f}s")

        # --- TensorBoard ---
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"{prefix_str}{k}", float(v), step)

        # --- WandB ---
        if self.wandb_active:
            wandb.log({f"{prefix_str}{k}": float(v) for k, v in metrics.items()}, step=step)

    def save_metrics(self, metrics_history: list, output_path: str):
        """
        Save list of metrics dicts over time to JSON.
        """
        with open(output_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"💾 Saved metrics history to {output_path}")

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_active:
            wandb.finish()
