import os
import matplotlib.pyplot as plt

def init_wandb(project_name, run_name=None, config=None):
    """
    Initialize a Weights & Biases (wandb) run.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is not installed. Run `pip install wandb` or use tensorboard instead.")
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
    )
    return run

def init_tensorboard(log_dir="runs"):
    """
    Initialize a TensorBoard log writer.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError("TensorBoard not available. Install with `pip install tensorboard torch`.")

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def log_metrics_wandb(metrics: dict, step: int):
    """
    Log a dictionary of scalar metrics to Weights & Biases.
    """
    import wandb
    wandb.log(metrics, step=step)

def log_metrics_tensorboard(writer, metrics: dict, step: int):
    """
    Log a dictionary of scalar metrics to TensorBoard.
    """
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)

def plot_losses(train_losses, val_losses=None, poincare_losses=None, val_steps=None, save_path="loss_graph.pdf"):
    """
    Plot and save a graph of training/validation/Poincaré loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")

    if val_losses is not None and val_steps is not None:
        plt.plot(val_steps, val_losses, 'ro', label="Validation Loss")

    if poincare_losses is not None and len(poincare_losses) == len(train_losses):
        plt.plot(poincare_losses, label="Poincaré Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training/Validation/Poincaré Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Saved loss plot to {save_path}")
