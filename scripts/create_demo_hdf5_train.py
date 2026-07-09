"""Create CTAT training HDF5 files from the bundled demo subject.

This is for formal-training smoke tests only. The output uses one subject, so it
does not provide paper-level training or validation evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.demo_ctat import load_subject_data
from scripts.train_single_subject_ctat import build_weight_maps


MODALITIES = (
    "FractionalAnisotropy",
    "Trace",
    "MinEigenvalue",
    "MidEigenvalue",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(ROOT / "testdata" / "HCP-100337-b1000"),
        help="Directory containing the bundled preprocessed demo subject.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "data" / "hdf5_train"),
        help="Output directory for four modality HDF5 files.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression. Use gzip to reduce disk usage.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, num_classes = load_subject_data(str(data_dir))
    if images.shape[1] != len(MODALITIES) * 7:
        raise ValueError(f"Expected 28 channels, got {images.shape[1]}")

    labels = labels.astype(np.int16, copy=False)
    weights = build_weight_maps(labels, num_classes).astype(np.float32, copy=False)
    subject = np.array([data_dir.name] * labels.shape[0], dtype=h5py.string_dtype())
    compression = None if args.compression == "none" else args.compression

    outputs = []
    for mod_idx, modality in enumerate(MODALITIES):
        start = mod_idx * 7
        stop = start + 7
        modality_images = images[:, start:stop, :, :]
        modality_images = np.transpose(modality_images, (0, 2, 3, 1)).astype(
            np.float32, copy=False
        )

        out_path = out_dir / f"demo_{modality}.hdf5"
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset(
                "orig_dataset",
                data=modality_images,
                compression=compression,
                chunks=(1, 256, 256, 7),
            )
            hf.create_dataset(
                "aseg_dataset",
                data=labels,
                compression=compression,
                chunks=(1, 256, 256),
            )
            hf.create_dataset(
                "weight_dataset",
                data=weights,
                compression=compression,
                chunks=(1, 256, 256),
            )
            hf.create_dataset("subject", data=subject)
        outputs.append(str(out_path))
        print(f"wrote {out_path}")

    manifest = {
        "source": str(data_dir),
        "purpose": "single-subject formal-training smoke test; not paper evidence",
        "num_slices": int(labels.shape[0]),
        "num_classes": int(num_classes),
        "modalities": list(MODALITIES),
        "outputs": outputs,
    }
    manifest_path = out_dir / "demo_hdf5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
