"""Compare a reproduced DDParcel segmentation with the bundled reference."""

from __future__ import annotations

import argparse

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default=r"testdata\HCP-100337-b1000\HCP-100337-b1000-DDSurfer-wmparc-Reg.mgz",
    )
    parser.add_argument(
        "--prediction",
        default=r"runs\baseline_demo\HCP-100337-b1000-DDSurfer-wmparc-Reg.mgz",
    )
    args = parser.parse_args()

    ref = np.asanyarray(nib.load(args.reference).dataobj)
    pred = np.asanyarray(nib.load(args.prediction).dataobj)

    same_shape = ref.shape == pred.shape
    if same_shape:
        voxel_agreement = float((ref == pred).mean())
        different_voxels = int((ref != pred).sum())
    else:
        voxel_agreement = float("nan")
        different_voxels = -1

    print(f"same_shape {same_shape}")
    print(f"ref_shape {ref.shape}")
    print(f"pred_shape {pred.shape}")
    print(f"voxel_agreement {voxel_agreement}")
    print(f"different_voxels {different_voxels}")
    print(f"ref_labels {len(np.unique(ref))}")
    print(f"pred_labels {len(np.unique(pred))}")


if __name__ == "__main__":
    main()
