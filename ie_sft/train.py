import argparse
from pathlib import Path

import torch

from dataset import build_dataloaders
from model import IEModel, ModelConfig
from trainer import TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for information extraction")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Base open-source model (<2B)")
    parser.add_argument("--data_path", type=str, default="ie_sft/data/sft_samples.json")
    parser.add_argument("--output_dir", type=str, default="ie_sft/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eval_every_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, val_loader, tokenizer = build_dataloaders(
        model_name=args.model_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model_wrapper = IEModel(ModelConfig(model_name=args.model_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        eval_every_steps=args.eval_every_steps,
        max_checkpoints=5,
    )

    metrics_history = train(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    if metrics_history:
        best = max(metrics_history, key=lambda x: x["precision"])
        print(f"Best precision={best['precision']:.4f} at step={best['step']}")
    else:
        print("No evaluation happened. Consider lowering --eval_every_steps.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
