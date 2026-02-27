import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from dataset import FIELD_KEYS


def _safe_json_loads(text: str) -> Dict[str, str]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def compute_precision(predictions: List[str], references: List[str]) -> float:
    correct = 0
    predicted_total = 0

    for pred, ref in zip(predictions, references):
        pred_dict = _safe_json_loads(pred)
        ref_dict = _safe_json_loads(ref)
        for key in FIELD_KEYS:
            if key in pred_dict:
                predicted_total += 1
                if pred_dict.get(key, "") == ref_dict.get(key, ""):
                    correct += 1

    if predicted_total == 0:
        return 0.0
    return correct / predicted_total


@torch.no_grad()
def evaluate_model(model, tokenizer, dataloader, device: torch.device, max_new_tokens: int = 128) -> Dict[str, float]:
    model.eval()
    preds, refs = [], []

    for batch in dataloader:
        raw_targets = batch.pop("raw_targets")
        batch.pop("raw_inputs")
        batch = {k: v.to(device) for k, v in batch.items()}

        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds.extend(decoded_preds)
        refs.extend(raw_targets)

    precision = compute_precision(preds, refs)
    return {"precision": precision}


def save_metrics_log(metrics_history: List[Dict[str, float]], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)


def plot_precision_curve(metrics_history: List[Dict[str, float]], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    steps = [item["step"] for item in metrics_history]
    values = [item["precision"] for item in metrics_history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker="o", linewidth=2)
    plt.title("Validation Precision Curve")
    plt.xlabel("Step")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
