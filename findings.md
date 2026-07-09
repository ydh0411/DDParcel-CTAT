# Findings: DDParcel-CTAT

## Project Summary
- `DDParcel-CTAT` is based on `zhangfanmark/DDParcel`, whose original paper is: Fan Zhang et al., "DDParcel: Deep Learning Anatomical Brain Parcellation From Diffusion MRI", IEEE TMI 2024, DOI `10.1109/TMI.2023.3331691`.
- The cloned CTAT project adds a Transformer-style model for modality-competitive token fusion while keeping much of DDParcel's preprocessing and three-view inference context.

## Environment
- Base Python at `D:\Annaconda\python.exe` does not have `torch`.
- `med_ai` and `med_ai_310` both have `torch 2.10.0+cu130` and CUDA available.
- `med_ai_310` currently does not have `pytest`.

## Data Sources
- DDParcel release assets provide the immediate demo data and pretrained DDParcel weights.
- Full paper datasets are separate:
  - HCP: `https://www.humanconnectome.org`
  - CNP: `https://openfmri.org/dataset/ds000030`
  - PPMI: `https://www.ppmi-info.org`
  - VERIO: in-house dataset, request only under data use agreement.

## Reproducibility Notes
- The demo release assets are enough for a first baseline smoke run.
- CTAT training needs HDF5 files ordered as `FractionalAnisotropy`, `Trace`, `MinEigenvalue`, `MidEigenvalue`.
- Subject-level splitting is required; never split train/val/test by slice.
- The single-subject CTAT run is an engineering validation only: it shows the model, loss, optimizer, CUDA path, and artifact saving work on real demo data, but it is not evidence for paper-level generalization.
- `data/hdf5_train` contains generated HDF5 files from the bundled demo subject. They are suitable for formal loader smoke tests only, not for full training claims.
