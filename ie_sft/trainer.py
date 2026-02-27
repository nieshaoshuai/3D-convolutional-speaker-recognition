import heapq
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from eval import evaluate_model, plot_loss_curve, plot_precision_curve, save_metrics_log


@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_every_steps: int = 10
    max_checkpoints: int = 5


class CheckpointManager:
    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_heap: List[Tuple[float, str, int]] = []

    def maybe_save(
        self,
        model,
        tokenizer,
        metric_value: float,
        step: int,
        metrics_history: List[Dict[str, float]],
    ):
        ckpt_name = f"step_{step}"
        ckpt_path = self.output_dir / ckpt_name

        if len(self.best_heap) < self.max_checkpoints:
            self._save_checkpoint(model, tokenizer, ckpt_path, step, metric_value)
            heapq.heappush(self.best_heap, (metric_value, ckpt_name, step))
            return

        worst_metric, worst_name, _ = self.best_heap[0]
        if metric_value <= worst_metric:
            return

        worst_path = self.output_dir / worst_name
        if worst_path.exists():
            shutil.rmtree(worst_path)

        heapq.heapreplace(self.best_heap, (metric_value, ckpt_name, step))
        self._save_checkpoint(model, tokenizer, ckpt_path, step, metric_value)

    def _save_checkpoint(self, model, tokenizer, ckpt_path: Path, step: int, metric_value: float):
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        with open(ckpt_path / "score.txt", "w", encoding="utf-8") as f:
            f.write(f"step={step}\nprecision={metric_value:.6f}\n")


def train(model_wrapper, tokenizer, train_loader, val_loader, config: TrainConfig, device: torch.device):
    model_wrapper.to(device)
    model = model_wrapper.model

    total_steps = len(train_loader) * config.epochs
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    checkpoint_manager = CheckpointManager(config.output_dir, config.max_checkpoints)
    metrics_history: List[Dict[str, float]] = []

    tensorboard_dir = Path(config.output_dir) / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    global_step = 0
    model.train()

    try:
        for epoch in range(config.epochs):
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
            for batch in progress:
                batch.pop("raw_inputs")
                batch.pop("raw_targets")
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                train_loss = loss.item()
                writer.add_scalar("loss/train_step", train_loss, global_step)
                progress.set_postfix({"loss": f"{train_loss:.4f}", "step": global_step})

                if global_step % config.eval_every_steps == 0:
                    metrics = evaluate_model(model_wrapper, tokenizer, val_loader, device)
                    metric_row = {
                        "step": global_step,
                        "train_loss": train_loss,
                        "eval_loss": metrics["eval_loss"],
                        "precision": metrics["precision"],
                    }
                    metrics_history.append(metric_row)

                    writer.add_scalar("loss/train_eval_step", train_loss, global_step)
                    writer.add_scalar("loss/eval", metrics["eval_loss"], global_step)
                    writer.add_scalar("metric/precision", metrics["precision"], global_step)

                    checkpoint_manager.maybe_save(
                        model=model,
                        tokenizer=tokenizer,
                        metric_value=metrics["precision"],
                        step=global_step,
                        metrics_history=metrics_history,
                    )

                    save_metrics_log(metrics_history, str(Path(config.output_dir) / "metrics.json"))
                    plot_precision_curve(metrics_history, str(Path(config.output_dir) / "precision_curve.png"))
                    plot_loss_curve(metrics_history, str(Path(config.output_dir) / "loss_curve.png"))
                    model.train()
    finally:
        writer.close()

    return metrics_history
