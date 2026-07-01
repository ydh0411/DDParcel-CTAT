#!/usr/bin/env python3
"""CTAT modality competition diagnostics — extract gate statistics and visualizations.

Usage:
    python scripts/diagnose_gate.py --ckpt experiments/ctat-coronal/best_model.pkl \
        --data_dir testdata/HCP-100337-b1000 --output results/gate_diagnostics/

Produces:
    - gate_entropy.npy          [N_patches] per-patch entropy over 4 modalities
    - winner_counts.npy         [4] global winner counts per modality
    - winner_map.png            spatial heatmap of dominant modality per patch
    - modality_selection.json   per-modality selection statistics
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import torch
import torch.nn.functional as F

from ctat_network import CTAT
from scripts.demo_ctat import load_subject_data


MODALITY_NAMES = ['FA', 'Trace', 'MinEig', 'MidEig']
FREESURFER_LABELS = {
    1: 'Background', 2: 'L-Caudate', 3: 'R-Caudate', 4: 'L-Putamen',
    5: 'R-Putamen', 6: 'L-Pallidum', 7: 'R-Pallidum', 8: 'L-Thalamus',
    9: 'R-Thalamus', 10: 'L-Hippocampus', 11: 'R-Hippocampus',
    12: 'L-Amygdala', 13: 'R-Amygdala', 14: 'L-Accumbens', 15: 'R-Accumbens',
    16: 'L-VentralDC', 17: 'R-VentralDC', 24: 'CSF',
    # Extended FreeSurfer labels — add more as needed
}


def compute_gate_statistics(gate, label_map=None):
    """Compute per-patch and per-modality gate statistics.

    Args:
        gate: [B, M, N, 1] from ModalityCompetitiveFusion.last_gate
        label_map: [B, N] optional anatomical label per patch

    Returns:
        dict with entropy, winner_counts, per_region_stats
    """
    gate = gate.squeeze(-1)  # [B, M, N]
    B, M, N = gate.shape

    # Per-patch entropy: -sum(p * log(p)) / log(M)
    eps = 1e-10
    entropy = -(gate * torch.log(gate + eps)).sum(dim=1) / np.log(M)  # [B, N]

    # Winner modality per patch
    winners = gate.argmax(dim=1)  # [B, N]
    winner_counts = torch.bincount(winners.flatten(), minlength=M).float()

    stats = {
        'entropy_mean': entropy.mean().item(),
        'entropy_std': entropy.std().item(),
        'winner_counts': {MODALITY_NAMES[i]: winner_counts[i].item() for i in range(M)},
        'winner_fractions': {MODALITY_NAMES[i]: (winner_counts[i] / winner_counts.sum()).item()
                            for i in range(M)},
    }

    # Per-region modality usage (if labels provided)
    if label_map is not None:
        region_stats = {}
        unique_labels = torch.unique(label_map).long()
        for lbl in unique_labels.tolist():
            if lbl == 0:
                continue
            mask = (label_map == lbl)
            if mask.sum() < 10:
                continue
            region_winners = winners.flatten()[mask.flatten()]
            region_counts = torch.bincount(region_winners, minlength=M).float()
            name = FREESURFER_LABELS.get(int(lbl), f'Label{lbl}')
            region_stats[name] = {
                'n_patches': mask.sum().item(),
                'winner': MODALITY_NAMES[region_counts.argmax().item()],
                'fractions': {MODALITY_NAMES[i]: (region_counts[i] / max(region_counts.sum(), 1)).item()
                              for i in range(M)},
            }
        stats['per_region'] = region_stats

    return stats, entropy, winners


def load_model(ckpt_path, num_classes, device):
    model = CTAT(
        num_classes=num_classes, in_channels=7, num_modalities=4,
        embed_dim=96, num_heads=8, depths=[2, 2, 2, 6], alpha=2.0,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    model.set_alpha(2.0)
    return model


def main():
    parser = argparse.ArgumentParser(description='CTAT modality gate diagnostics')
    parser.add_argument('--ckpt', help='Model checkpoint (.pkl)')
    parser.add_argument('--data_dir', required=True, help='Subject data directory')
    parser.add_argument('--output', default='./gate_diagnostics', help='Output directory')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num_classes', type=int, default=82)
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = torch.device(args.device)

    os.makedirs(args.output, exist_ok=True)

    # Load data
    images, labels, num_classes = load_subject_data(args.data_dir)
    images_t = torch.from_numpy(images).float().to(device)
    labels_t = torch.from_numpy(labels).long()

    # Load or create model
    if args.ckpt:
        model = load_model(args.ckpt, num_classes, device)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        model = CTAT(
            num_classes=num_classes, in_channels=7, num_modalities=4,
            embed_dim=96, num_heads=8, depths=[2, 2, 2, 6], alpha=2.0,
        ).to(device)
        model.eval()
        model.set_alpha(2.0)
        print("Using untrained model (random weights) — gate patterns are not meaningful")

    # Run inference on a subset of slices
    n_slices = min(32, len(images))
    indices = np.linspace(0, len(images) - 1, n_slices, dtype=int)

    all_entropy = []
    all_winners = []
    all_labels = []

    with torch.no_grad():
        for i in indices:
            x = images_t[i:i+1]
            _ = model(x, return_aux=False)

            gate = model.encoder.last_modality_gate  # [1, M, N, 1]
            if gate is None:
                print("ERROR: encoder.last_modality_gate is None. "
                      "Ensure ModalityCompetitiveFusion ran.")
                sys.exit(1)

            # Downsample labels to patch grid (64x64)
            label_slice = labels_t[i]  # [256, 256]
            label_patch = F.avg_pool2d(
                label_slice.float().unsqueeze(0).unsqueeze(0),
                kernel_size=4, stride=4,
            ).squeeze().round().long()  # [64, 64]

            stats, entropy, winners = compute_gate_statistics(
                gate.cpu(), label_patch.flatten().unsqueeze(0))
            all_entropy.append(entropy.squeeze(0).numpy())  # [N]
            all_winners.append(winners.squeeze(0).numpy())  # [N]
            all_labels.append(label_patch.flatten().numpy())  # [N]

    # Aggregate statistics
    entropy_all = np.concatenate(all_entropy)
    winners_all = np.concatenate(all_winners)
    labels_all = np.concatenate(all_labels)

    # Global stats
    global_winner_counts = np.bincount(winners_all, minlength=4)
    global_stats = {
        'model': args.ckpt or 'untrained',
        'n_patches_analyzed': len(winners_all),
        'gate_entropy': {'mean': float(entropy_all.mean()), 'std': float(entropy_all.std())},
        'winner_fractions': {MODALITY_NAMES[i]: float(global_winner_counts[i] / global_winner_counts.sum())
                            for i in range(4)},
        'active_modalities_per_patch': {
            '1_modality': float((entropy_all < 0.15).mean()),
            '2_modalities': float(((entropy_all >= 0.15) & (entropy_all < 0.5)).mean()),
            '3_modalities': float(((entropy_all >= 0.5) & (entropy_all < 0.85)).mean()),
            'all_modalities': float((entropy_all >= 0.85).mean()),
        },
    }

    # Save
    np.save(os.path.join(args.output, 'gate_entropy.npy'), entropy_all)
    np.save(os.path.join(args.output, 'winner_indices.npy'), winners_all)
    with open(os.path.join(args.output, 'gate_statistics.json'), 'w') as f:
        json.dump(global_stats, f, indent=2)

    print("\n=== CTAT Modality Gate Diagnostics ===")
    print(f"Patches analyzed: {global_stats['n_patches_analyzed']}")
    print(f"Gate entropy: {global_stats['gate_entropy']['mean']:.3f} +/- {global_stats['gate_entropy']['std']:.3f}")
    print(f"\nModality winner fractions:")
    for mod, frac in global_stats['winner_fractions'].items():
        bar = '█' * int(frac * 40) + '░' * (40 - int(frac * 40))
        print(f"  {mod:10s}: {frac:.3f} {bar}")
    print(f"\nSparsity breakdown:")
    for level, frac in global_stats['active_modalities_per_patch'].items():
        print(f"  {level:20s}: {frac:.3f}")
    print(f"\nResults saved to {args.output}/")


if __name__ == '__main__':
    main()
