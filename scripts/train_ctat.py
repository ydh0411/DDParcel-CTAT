# scripts/train_ctat.py
"""Train CTAT for one view (axial, coronal, or sagittal).

Uses subject-level train/val splits: all slices from a given subject stay together
in either train or val, never both. This prevents data leakage through slice adjacency.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from models.ctat_network import CTAT
from models.ctat_solver import CTATSolver
from data_loader.load_neuroimaging_data import AsegDatasetWithAugmentation_Fused_Input
from data_loader.augmentation import ToTensor, AugmentationPadImage, AugmentationRandomCrop


MODALITY_ORDER = ['FractionalAnisotropy', 'Trace', 'MinEigenvalue', 'MidEigenvalue']


def order_modality_files(hdf5_files):
    """Return HDF5 files in the same modality order used by inference."""
    ordered = []
    for modality in MODALITY_ORDER:
        matches = [path for path in hdf5_files if modality in os.path.basename(path)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one HDF5 file containing {modality}, found {len(matches)}")
        ordered.append(matches[0])
    return ordered


def build_dataset(hdf5_dir, view='coronal'):
    """Build the full AsegDatasetWithAugmentation_Fused_Input dataset."""
    import glob
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, '*.hdf5')))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_dir}")
    hdf5_files = order_modality_files(hdf5_files)

    params = {
        'dataset_name': hdf5_files,
        'plane': view.capitalize(),
    }
    dataset_transforms = transforms.Compose([
        AugmentationPadImage(),
        AugmentationRandomCrop(output_size=(256, 256)),
        ToTensor(),
    ])
    return AsegDatasetWithAugmentation_Fused_Input(params, transforms=dataset_transforms)


def subject_level_split(dataset, val_split=0.1, seed=42):
    """Split dataset by unique subjects, returning (train_indices, val_indices).

    All slices belonging to the same subject ID stay in the same split.
    Saves a split manifest to the experiment directory for reproducibility.
    """
    subjects = [
        s.decode('utf-8') if isinstance(s, bytes) else str(s)
        for s in dataset.get_subject_names()
    ]
    unique_subjects = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)

    if val_split <= 0 or len(unique_subjects) == 1:
        n_val = 0
    else:
        n_val = max(1, int(len(unique_subjects) * val_split))
        n_val = min(n_val, len(unique_subjects) - 1)
    n_train = len(unique_subjects) - n_val
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]

    print(f"Subject-level split: {len(unique_subjects)} subjects "
          f"-> {n_train} train ({len(train_idx)} slices) + "
          f"{n_val} val ({len(val_idx)} slices)")

    if not train_idx:
        raise ValueError("Subject-level split produced an empty training set")

    manifest = {
        'n_subjects_total': len(unique_subjects),
        'n_train_subjects': n_train,
        'n_val_subjects': n_val,
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'seed': seed,
        'val_split': val_split,
    }
    return train_idx, val_idx, manifest


def get_train_loader(hdf5_dir, view='coronal', batch_size=8, num_workers=4):
    """Build training DataLoader from HDF5 files."""
    dataset = build_dataset(hdf5_dir, view)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', required=True, help='Directory with training HDF5 files')
    parser.add_argument('--view', default='coronal', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--num_classes', type=int, default=82)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--exp_dir', default='./experiments/ctat')
    parser.add_argument('--alpha_start', type=float, default=1.0)
    parser.add_argument('--alpha_end', type=float, default=2.0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--max_train_slices',
        type=int,
        default=None,
        help='Optional cap for quick smoke tests; leave unset for full training.',
    )
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print(f"Training CTAT on {args.view} view, device={device}, seed={args.seed}")

    full_dataset = build_dataset(args.hdf5_dir, args.view)
    train_idx, val_idx, split_manifest = subject_level_split(
        full_dataset, args.val_split, args.seed)
    if args.max_train_slices is not None:
        train_idx = train_idx[:args.max_train_slices]
        val_idx = val_idx[:args.max_train_slices] if val_idx else val_idx
        split_manifest['max_train_slices'] = args.max_train_slices
        split_manifest['n_train_slices_used'] = len(train_idx)
        split_manifest['n_val_slices_used'] = len(val_idx)

    exp_dir = f"{args.exp_dir}-{args.view}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'split_manifest.json'), 'w') as f:
        json.dump(split_manifest, f, indent=2)

    train_loader = DataLoader(Subset(full_dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if val_idx:
        val_loader = DataLoader(Subset(full_dataset, val_idx),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    model = CTAT(
        num_classes=args.num_classes,
        in_channels=7,
        num_modalities=4,
        embed_dim=args.embed_dim,
        alpha=args.alpha_start,
    )

    solver = CTATSolver(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        total_epochs=args.epochs,
        device=device,
        exp_dir=exp_dir,
    )

    solver.train()
    print(f"Training complete. Model saved to {exp_dir}")


if __name__ == '__main__':
    main()
