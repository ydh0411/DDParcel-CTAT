"""Single-subject CTAT training smoke experiment.

This script is for engineering verification only. It trains CTAT on one bundled
HCP demo subject and saves reproducible run artifacts. Do not treat these
numbers as paper evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from models.ctat_network import CTAT
from models.ctat_solver import CombinedLoss
from scripts.demo_ctat import load_subject_data


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_weight_maps(labels_np: np.ndarray, num_classes: int) -> np.ndarray:
    class_counts = np.bincount(labels_np.reshape(-1), minlength=num_classes)
    present = class_counts > 0
    median_freq = np.median(class_counts[present])
    class_weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx in range(num_classes):
        if class_counts[cls_idx] > 0:
            class_weights[cls_idx] = np.clip(
                median_freq / class_counts[cls_idx], 0.1, 50.0
            )

    weight_maps = np.zeros_like(labels_np, dtype=np.float32)
    for cls_idx, cls_weight in enumerate(class_weights):
        weight_maps[labels_np == cls_idx] = cls_weight
    return weight_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(ROOT / "testdata" / "HCP-100337-b1000"),
        help="Directory containing one preprocessed HCP subject.",
    )
    parser.add_argument(
        "--exp_dir",
        default=str(ROOT / "experiments" / "2026-07-09_single-subject-ctat"),
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--max_slices", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    set_seed(args.seed, device)

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["purpose"] = "single-subject engineering smoke test; not paper evidence"
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    started = time.time()
    images, labels, num_classes = load_subject_data(args.data_dir)
    n_total = len(images)
    n_train = min(n_total, args.max_slices)
    slice_indices = np.linspace(0, n_total - 1, n_train, dtype=int)

    images_np = images[slice_indices]
    labels_np = labels[slice_indices]
    weights_np = build_weight_maps(labels_np, num_classes)

    images_t = torch.from_numpy(images_np).float().to(device)
    labels_t = torch.from_numpy(labels_np).long().to(device)
    weights_t = torch.from_numpy(weights_np).float().to(device)

    model = CTAT(
        num_classes=num_classes,
        in_channels=7,
        num_modalities=4,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
        depths=[2, 2, 2, 4],
        alpha=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=1e-6
    )
    loss_fn = CombinedLoss(weight_dice=1, weight_ce=1)
    ds_weights = [0.25, 0.5, 0.75]

    n_params = sum(param.numel() for param in model.parameters())
    print(f"device {device}")
    print(f"classes {num_classes}")
    print(f"slices {n_train}")
    print(f"params {n_params / 1e6:.1f}M")
    print("iter loss main dice ce alpha lr")

    rows = []
    losses = []
    model.train()
    for iteration in range(args.iterations):
        rng = np.random.RandomState(args.seed + iteration)
        batch_idx = rng.choice(n_train, args.batch_size, replace=False)

        x = images_t[batch_idx]
        y = labels_t[batch_idx]
        w = weights_t[batch_idx]

        alpha = 1.0 + min(iteration / max(args.iterations, 1), 0.95)
        model.set_alpha(alpha)

        optimizer.zero_grad()
        main_logits, aux_logits = model(x, return_aux=True)
        main_loss, dice_loss, ce_loss = loss_fn(main_logits, y, w)
        loss = main_loss
        for aux_idx, aux_logit in enumerate(aux_logits):
            aux_loss, _, _ = loss_fn(aux_logit, y, w)
            loss = loss + ds_weights[aux_idx] * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        row = {
            "iteration": iteration + 1,
            "loss": loss_value,
            "main_loss": float(main_loss.item()),
            "dice_loss": float(dice_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "alpha": float(alpha),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)

        if iteration == 0 or (iteration + 1) % args.log_every == 0:
            print(
                f"{iteration + 1} {row['loss']:.4f} {row['main_loss']:.4f} "
                f"{row['dice_loss']:.4f} {row['ce_loss']:.4f} "
                f"{row['alpha']:.2f} {row['lr']:.2e}"
            )

    with (exp_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_change_percent": (1 - losses[-1] / losses[0]) * 100,
        "iterations": args.iterations,
        "num_classes": num_classes,
        "num_slices": n_train,
        "num_parameters": n_params,
        "elapsed_seconds": time.time() - started,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        exp_dir / "final_model.pkl",
    )

    print(f"initial_loss {metrics['initial_loss']:.4f}")
    print(f"final_loss {metrics['final_loss']:.4f}")
    print(f"loss_change_percent {metrics['loss_change_percent']:.1f}")
    print(f"saved {exp_dir}")


if __name__ == "__main__":
    main()
