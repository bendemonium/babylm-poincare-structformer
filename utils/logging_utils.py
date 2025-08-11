import os
import sys
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import numpy as np

# Optional wandb integration
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    log_dir: str = "logs/structformer_run"
    use_wandb: bool = True
    wandb_project: str = "structformer-flax"
    wandb_run_name: str = "run-001"
    wandb_entity: Optional[str] = None

    # Console/file logging
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    # Metric logging intervals
    log_interval_steps: int = 100
    log_interval_words: int = 100_000
    eval_log_interval: int = 1_000

    # Progress tracking
    progress_bar: bool = True
    words_per_second_window: int = 1000  # number of recent batches

    # Alerts
    loss_spike_threshold: float = 2.0
    nan_alert: bool = True
    memory_alert_threshold_gb: float = 8.0


def parse_logging_config(config_dict: Dict) -> LoggingConfig:
    """Parse logging config from main config dictionary."""
    lc = config_dict.get("logging", {})
    return LoggingConfig(
        log_dir=lc.get("log_dir", "logs/structformer_run"),
        use_wandb=lc.get("use_wandb", True),
        wandb_project=lc.get("wandb_project", "structformer-flax"),
        wandb_run_name=lc.get("wandb_run_name", "run-001"),
        wandb_entity=lc.get("wandb_entity"),
        console_log_level=lc.get("console_log_level", "INFO"),
        file_log_level=lc.get("file_log_level", "DEBUG"),
        log_interval_steps=lc.get("log_interval_steps", 100),
        log_interval_words=lc.get("log_interval_words", 100_000),
        eval_log_interval=lc.get("eval_log_interval", 1_000),
        progress_bar=lc.get("progress_bar", True),
        words_per_second_window=lc.get("words_per_second_window", 1000),
        loss_spike_threshold=lc.get("loss_spike_threshold", 2.0),
        nan_alert=lc.get("nan_alert", True),
        memory_alert_threshold_gb=lc.get("memory_alert_threshold_gb", 8.0),
    )

# ----------------------------
# Logging setup
# ----------------------------
def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Set up logging with both console and file handlers."""
    os.makedirs(config.log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, config.console_log_level.upper()))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(config.log_dir, "training.log"), encoding="utf-8")
    fh.setLevel(getattr(logging, config.file_log_level.upper()))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    main_logger = logging.getLogger("structformer_training")
    main_logger.info("✅ Logging setup complete. Logs: %s", os.path.join(config.log_dir, "training.log"))
    return main_logger


def setup_wandb(config: LoggingConfig, model_config: Any = None) -> bool:
    """Initialize Weights & Biases logging."""
    if not config.use_wandb or not WANDB_AVAILABLE:
        logger.info("W&B disabled or not available")
        return False
    try:
        wb_cfg = {}
        if model_config is not None:
            wb_cfg.update({
                "model_type": "structformer_poincare",
                "framework": "jax_flax",
                "vocab_size": getattr(model_config, "vocab_size", None),
                "hidden_dim": getattr(model_config, "hidden_dim", None),
                "num_layers": getattr(model_config, "num_layers", None),
                "num_heads": getattr(model_config, "num_heads", None),
                "max_length": getattr(model_config, "max_length", None),
                "hyperbolic_c": getattr(model_config, "c", None),
            })
            if hasattr(model_config, "training"):
                tr = model_config.training
                wb_cfg.update({
                    "lr_ce": getattr(tr, "lr_ce", None),
                    "lr_riem": getattr(tr, "lr_riem", None),
                    "lambda_poincare": getattr(tr, "lambda_poincare", None),
                    "batch_size": getattr(tr, "batch_size", None),
                    "max_words": getattr(tr, "max_words", None),
                })

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config=wb_cfg,
            save_code=True,
            dir=config.log_dir,
        )
        logger.info("✅ W&B initialized: %s/%s", config.wandb_project, config.wandb_run_name)
        return True
    except Exception as e:
        logger.error("Failed to initialize W&B: %s", str(e))
        return False

# ----------------------------
# Metrics tracking
# ----------------------------
class MetricsTracker:
    """Track and log training metrics with moving averages and alerts."""
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.step_times = deque(maxlen=config.words_per_second_window)
        self.step_words = deque(maxlen=config.words_per_second_window)

        self.start_time = time.time()
        self.last_log_step = 0
        self.last_log_words = 0
        self.last_eval_step = 0

        self.last_loss_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.alert_counts: Dict[str, int] = defaultdict(int)

    def update_step_timing(self, step: int, words_in_batch: int):
        now = time.time()
        self.step_times.append(now)
        self.step_words.append(int(words_in_batch))

    def get_throughput_metrics(self) -> Dict[str, float]:
        if len(self.step_times) < 2:
            return {}
        total_time = self.step_times[-1] - self.step_times[0]
        total_words = int(sum(self.step_words))
        if total_time <= 0:
            return {}
        return {
            "words_per_second": float(total_words / total_time),
            "tokens_per_second": float(total_words / total_time),
            "time_per_batch": float(total_time / len(self.step_words)),
        }

    def _to_scalar(self, v: Any) -> Optional[float]:
        try:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                if v.size == 1:
                    return float(v.reshape(()).item())
                return None
            if hasattr(v, "item"):
                return float(v.item())
            return None
        except Exception:
            return None

    def add_metrics(self, metrics: Dict[str, Union[float, jnp.ndarray]], step: int):
        for k, v in metrics.items():
            s = self._to_scalar(v)
            if s is not None and np.isfinite(s):
                self.metrics_history[k].append(float(s))

    def get_smoothed_metrics(self, window: int = 100) -> Dict[str, float]:
        out = {}
        for k, dq in self.metrics_history.items():
            if len(dq) == 0:
                continue
            recent = list(dq)[-window:]
            out[f"{k}_smooth"] = float(np.mean(recent))
            out[f"{k}_current"] = float(recent[-1])
        return out

    def check_alerts(self, metrics: Dict[str, float], step: int) -> List[str]:
        alerts: List[str] = []

        # NaN / Inf
        if self.config.nan_alert:
            for k, v in metrics.items():
                s = self._to_scalar(v)
                if s is not None and (not np.isfinite(s)):
                    alerts.append(f"🚨 NaN/Inf detected in {k} at step {step}")

        # Loss spikes
        for k, v in metrics.items():
            if "loss" not in k.lower():
                continue
            s = self._to_scalar(v)
            if s is None or not np.isfinite(s):
                continue
            self.last_loss_values[k].append(s)
            if len(self.last_loss_values[k]) >= 10:
                recent = list(self.last_loss_values[k])[-5:]
                older = list(self.last_loss_values[k])[:-5] or recent
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                if older_avg > 0 and recent_avg / older_avg > self.config.loss_spike_threshold:
                    alerts.append(f"⚠️ Loss spike in {k}: {older_avg:.4f} → {recent_avg:.4f}")

        # Memory alert (best-effort)
        try:
            import psutil  # type: ignore
            mem_gb = psutil.virtual_memory().used / (1024 ** 3)
            if mem_gb > self.config.memory_alert_threshold_gb:
                alerts.append(f"🔥 High memory usage: {mem_gb:.1f}GB")
        except Exception:
            pass

        return alerts

    def should_log_step(self, step: int) -> bool:
        return (step - self.last_log_step) >= self.config.log_interval_steps

    def should_log_words(self, words_processed: int) -> bool:
        return (words_processed - self.last_log_words) >= self.config.log_interval_words

    def should_eval_log(self, step: int) -> bool:
        return (step - self.last_eval_step) >= self.config.eval_log_interval

# ----------------------------
# Progress monitoring
# ----------------------------
class ProgressMonitor:
    """Monitor training progress with word-based milestones."""
    def __init__(self, max_words: int, config: LoggingConfig):
        self.max_words = max_words
        self.config = config
        self.start_time = time.time()
        self.milestone_words = self._generate_milestones()
        self.completed_milestones = set()

    def _generate_milestones(self) -> List[int]:
        # 1M, 2M, 5M, 10M, 20M, 50M, 100M (capped by max_words)
        ms = [m * 1_000_000 for m in (1, 2, 5, 10, 20, 50, 100)]
        return [w for w in ms if w <= self.max_words]

    def update_progress(self, words_processed: int, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        progress_pct = float(min(100.0, (words_processed / max(1, self.max_words)) * 100.0))

        if words_processed > 0 and elapsed > 0:
            wps = words_processed / elapsed
            remaining_words = max(0, self.max_words - words_processed)
            eta = remaining_words / max(wps, 1e-8)
        else:
            wps, eta = 0.0, 0.0

        new_ms = []
        for m in self.milestone_words:
            if m <= words_processed and m not in self.completed_milestones:
                new_ms.append(m)
                self.completed_milestones.add(m)

        return {
            "words_processed": int(words_processed),
            "max_words": int(self.max_words),
            "progress_pct": progress_pct,
            "elapsed_time": float(elapsed),
            "estimated_remaining": float(eta),
            "words_per_second": float(wps),
            "new_milestones": new_ms,
            "completed_milestones": len(self.completed_milestones),
            "total_milestones": len(self.milestone_words),
        }

    def format_time(self, seconds: float) -> str:
        seconds = float(max(0.0, seconds))
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds/60:.1f}m"
        return f"{seconds/3600:.1f}h"

# ----------------------------
# Main logging interface
# ----------------------------
class TrainingLogger:
    """Main logging interface combining all logging utilities."""
    def __init__(self, config: LoggingConfig, model_config: Any = None, max_words: int = 100_000_000):
        self.config = config
        self.model_config = model_config

        self.logger = setup_logging(config)
        self.wandb_enabled = setup_wandb(config, model_config)
        self.metrics_tracker = MetricsTracker(config)
        self.progress_monitor = ProgressMonitor(max_words, config)

        self.training_start_time = time.time()
        self.last_checkpoint_words = 0

    def log_training_step(
        self,
        step: int,
        words_processed: int,
        metrics: Dict[str, Any],
        words_in_batch: int = 0,
        force_log: bool = False,
    ):
        # Timing + metrics
        self.metrics_tracker.update_step_timing(step, words_in_batch)
        self.metrics_tracker.add_metrics(metrics, step)

        should_step = self.metrics_tracker.should_log_step(step) or force_log
        should_words = self.metrics_tracker.should_log_words(words_processed) or force_log
        if not (should_step or should_words):
            return

        smoothed = self.metrics_tracker.get_smoothed_metrics()
        throughput = self.metrics_tracker.get_throughput_metrics()
        progress = self.progress_monitor.update_progress(words_processed, metrics)

        # Alerts
        for alert in self.metrics_tracker.check_alerts(metrics, step):
            self.logger.warning(alert)

        # Console
        if should_words or (step % max(1, self.config.log_interval_steps * 10) == 0):
            self._log_progress_console(step, words_processed, metrics, progress, throughput)

        # W&B
        if self.wandb_enabled:
            self._log_to_wandb(step, words_processed, {**metrics, **smoothed, **throughput, **progress})

        # File (jsonl)
        self._log_detailed_metrics(step, words_processed, metrics, smoothed, throughput)

        if should_step:
            self.metrics_tracker.last_log_step = step
        if should_words:
            self.metrics_tracker.last_log_words = words_processed

    def log_evaluation(self, step: int, words_processed: int, eval_metrics: Dict[str, Any]):
        self.logger.info("📊 Evaluation at step %s (%s words)", f"{step:,}", f"{words_processed:,}")
        for k, v in eval_metrics.items():
            s = v
            try:
                if hasattr(v, "item"):
                    s = float(v.item())
                elif isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                    s = float(v.reshape(()).item())
                elif isinstance(v, (int, float)):
                    s = float(v)
                self.logger.info("  %s: %.6f", k, s)
            except Exception:
                self.logger.info("  %s: %s", k, str(v))

        if self.wandb_enabled:
            to_log = {f"eval/{k}": (float(v) if isinstance(v, (int, float)) else v) for k, v in eval_metrics.items()}
            to_log.update({"step": step, "words_processed": words_processed})
            try:
                wandb.log(to_log)
            except Exception as e:
                self.logger.warning("W&B eval log failed: %s", str(e))

        self.metrics_tracker.last_eval_step = step

    def log_hyperbolic_diagnostics(self, step: int, words_processed: int, diagnostics: Dict[str, float]):
        self.logger.debug("🌀 Hyperbolic diagnostics at step %s", f"{step:,}")
        for k, v in diagnostics.items():
            try:
                self.logger.debug("  %s: %.6f", k, float(v))
            except Exception:
                self.logger.debug("  %s: %s", k, str(v))

        if self.wandb_enabled:
            payload = {f"hyperbolic/{k}": float(v) if isinstance(v, (int, float)) else v for k, v in diagnostics.items()}
            payload.update({"step": step, "words_processed": words_processed})
            try:
                wandb.log(payload)
            except Exception as e:
                self.logger.warning("W&B hyperbolic log failed: %s", str(e))

    def log_checkpoint_save(self, step: int, words_processed: int, checkpoint_path: str):
        name = self._get_milestone_name(words_processed)
        self.logger.info("💾 Checkpoint saved: %s (%s)", name, checkpoint_path)
        if self.wandb_enabled:
            try:
                wandb.log({
                    "checkpoint/words_processed": words_processed,
                    "checkpoint/step": step,
                    "step": step,
                    "words_processed": words_processed,
                })
            except Exception:
                pass
        self.last_checkpoint_words = words_processed

    def log_milestone(self, words_processed: int, milestone_metrics: Dict[str, Any] = None):
        name = self._get_milestone_name(words_processed)
        elapsed = time.time() - self.training_start_time
        self.logger.info("🎯 Milestone reached: %s", name)
        self.logger.info("   Elapsed time: %s", self.progress_monitor.format_time(elapsed))
        if milestone_metrics:
            self.logger.info("   Metrics at milestone:")
            for k, v in milestone_metrics.items():
                try:
                    if hasattr(v, "item"):
                        v = float(v.item())
                    self.logger.info("     %s: %.6f", k, float(v))
                except Exception:
                    self.logger.info("     %s: %s", k, str(v))

    def _log_progress_console(self, step: int, words_processed: int,
                              metrics: Dict[str, Any],
                              progress: Dict[str, Any],
                              throughput: Dict[str, Any]):
        def fmt_num(x):
            try:
                return float(x)
            except Exception:
                return None

        parts = []
        tl = fmt_num(metrics.get("total_loss"))
        ce = fmt_num(metrics.get("ce_loss"))
        hl = fmt_num(metrics.get("poincare_loss"))
        if tl is not None: parts.append(f"Loss={tl:.4f}")
        if ce is not None: parts.append(f"CE={ce:.4f}")
        if hl is not None: parts.append(f"Poincaré={hl:.4f}")

        wps = fmt_num(throughput.get("words_per_second"))
        if wps is not None: parts.append(f"{wps:.0f}w/s")

        prog = f"{progress.get('progress_pct', 0.0):.1f}%"
        eta_s = progress.get("estimated_remaining", 0.0)
        if eta_s and eta_s > 0:
            prog += f", ETA {self.progress_monitor.format_time(eta_s)}"

        self.logger.info(
            "Step %s | %sK words | %s | %s",
            f"{step:,}",
            f"{words_processed//1000:,}",
            prog,
            " | ".join(parts) if parts else "no-metrics",
        )

    def _log_to_wandb(self, step: int, words_processed: int, log_dict: Dict[str, Any]):
        filtered: Dict[str, Any] = {}
        for k, v in log_dict.items():
            try:
                if isinstance(v, (int, float, bool, str)):
                    filtered[k] = v
                elif hasattr(v, "item"):
                    filtered[k] = float(v.item())
                elif isinstance(v, (jnp.ndarray, np.ndarray)) and getattr(v, "size", 0) == 1:
                    filtered[k] = float(v.reshape(()).item())
            except Exception:
                continue

        filtered.update({"step": step, "words_processed": words_processed})
        try:
            wandb.log(filtered)
        except Exception as e:
            self.logger.warning("W&B log failed: %s", str(e))

    def _log_detailed_metrics(self, step: int, words_processed: int,
                              metrics: Dict[str, Any],
                              smoothed: Dict[str, Any],
                              throughput: Dict[str, Any]):
        def to_floatable(v):
            try:
                if hasattr(v, "item"):
                    return float(v.item())
                if isinstance(v, (jnp.ndarray, np.ndarray)) and v.size == 1:
                    return float(v.reshape(()).item())
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                pass
            return v

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "words_processed": words_processed,
            "metrics": {k: to_floatable(v) for k, v in metrics.items()},
            "smoothed_metrics": smoothed,
            "throughput_metrics": throughput,
        }
        path = os.path.join(self.config.log_dir, "detailed_metrics.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")

    def _get_milestone_name(self, words_processed: int) -> str:
        if words_processed >= 1_000_000:
            return f"{words_processed//1_000_000}M words"
        if words_processed >= 1_000:
            return f"{words_processed//1_000}K words"
        return f"{words_processed} words"

    def finalize(self):
        total = time.time() - self.training_start_time
        self.logger.info("🏁 Training completed in %s", self.progress_monitor.format_time(total))
        if self.wandb_enabled and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception:
                pass

# ----------------------------
# Utility functions
# ----------------------------
def create_training_logger(config_dict: Dict, model_config: Any = None, max_words: int = 100_000_000) -> TrainingLogger:
    """Create training logger from configuration."""
    return TrainingLogger(parse_logging_config(config_dict), model_config, max_words)

def log_model_summary(tlogger: TrainingLogger, model, params, sample_input: Dict[str, Any]):
    """Log a quick model summary (param count and key output shapes)."""
    try:
        param_leaves = jax.tree_util.tree_leaves(params)
        param_count = int(sum(int(x.size) for x in param_leaves))
        sample_out = model.apply({"params": params}, **sample_input, training=False)

        tlogger.logger.info("🏗️  Model Architecture Summary")
        tlogger.logger.info("   Total parameters: %s", f"{param_count:,}")
        if isinstance(sample_out, dict):
            for k, v in sample_out.items():
                if hasattr(v, "shape"):
                    tlogger.logger.info("   %s shape: %s", k, tuple(v.shape))
        elif hasattr(sample_out, "shape"):
            tlogger.logger.info("   Output shape: %s", tuple(sample_out.shape))

        if tlogger.wandb_enabled:
            try:
                wandb.log({
                    "model/param_count": param_count,
                    "model/param_count_millions": param_count / 1_000_000.0,
                })
            except Exception:
                pass
    except Exception as e:
        tlogger.logger.warning("Failed to log model summary: %s", str(e))