# Progress Log: DDParcel-CTAT

## Session: 2026-07-09

### Clone and Orientation
- Cloned `https://github.com/ydh0411/DDParcel-CTAT` into `D:\codex-test\DDParcel-CTAT`.
- Deleted accidental duplicate clone at `D:\codex-test\external\DDParcel-CTAT` after user approval.
- Fetched original DDParcel baseline into local ref `upstream/main`.
- Compared CTAT against upstream DDParcel: 28 files changed, 6579 insertions, 938 deletions.

### Environment Check
- Base Python does not have `torch`.
- `med_ai_310` has `torch 2.10.0+cu130`; CUDA is available.
- `med_ai_310` does not have `pytest`.
- Direct execution of the eight functions in `tests/test_ctat_core.py` under `med_ai_310` passed.

### Current Task
- User requested downloading data and running experiments one by one.
- Created project-local planning files: `task_plan.md`, `findings.md`, `progress.md`.

### Public Asset Download Attempt
- Downloaded `100HCP-population-mean-T2-1mm.nii.gz` successfully; local size is `13,803,987` bytes.
- `weights.zip` download via `Invoke-WebRequest` timed out after 15 minutes and left a partial file of `70,704,272` bytes.
- Retried `weights.zip` with `curl.exe -L -C -` for resume; connection reset with `curl: (56) Recv failure: Connection was reset`.
- User chose to download the assets manually from links.

### Public Asset Folder Verification
- User clarified the transferred assets are folders, not zip files.
- Verified project root contains `weights/`, `testdata/`, and `100HCP-population-mean-T2-1mm.nii.gz`.
- Verified `weights/` contains 3 fused model checkpoints and `weights/backbones/` contains 12 backbone checkpoints.
- Verified `testdata/` contains 36 files with total size `612,108,306` bytes.
- Verified four normalized registered DTI inputs exist under `testdata/HCP-100337-b1000/`:
  - `FractionalAnisotropy`
  - `MidEigenvalue`
  - `MinEigenvalue`
  - `Trace`

### Baseline Demo Run - Transform Wrapper Fix
- User started baseline inference with:
  - `python .\DDSurfer_Pred.py --in_dir .\testdata\HCP-100337-b1000 --out_dir .\runs\baseline_demo --weights_dir .\weights`
- Run reached axial model loading on CUDA, then failed in the DataLoader with:
  - `TypeError: 'ToTensorTest' object is not iterable`
- Root cause:
  - `DDSurfer_Pred.py` used `transforms.Compose(ToTensorTest())` instead of `transforms.Compose([ToTensorTest()])`.
- Fixed `DDSurfer_Pred.py` line in `run_network`.
- Minimal verification:
  - `conda run -n med_ai_310 python -c "from torchvision import transforms; from data_loader.augmentation import ToTensorTest; t=transforms.Compose([ToTensorTest()]); print(type(t));"`
  - Output confirmed `torchvision.transforms.transforms.Compose`.

### Baseline Demo Verification
- User reran baseline inference after the transform fix.
- Output file:
  - `runs/baseline_demo/HCP-100337-b1000-DDSurfer-wmparc-Reg.mgz`
  - Size: `584,010` bytes
  - Timestamp: `2026-07-09 20:12:07`
- Verified with `nibabel` under `med_ai_310`:
  - Shape: `(256, 256, 256)`
  - dtype: `>i2`
  - min/max label values: `0` / `2035`
  - unique labels: `102`
- Phase 3 baseline demo is complete; next step is result comparison against the bundled reference output and CTAT smoke tests.

### Baseline Reproducibility Comparison
- User ran `python .\scripts\compare_baseline.py`.
- Comparison result:
  - `same_shape True`
  - `ref_shape (256, 256, 256)`
  - `pred_shape (256, 256, 256)`
  - `voxel_agreement 1.0`
  - `different_voxels 0`
  - `ref_labels 102`
  - `pred_labels 102`
- The baseline DDParcel demo reproduced the bundled reference output exactly.

### CTAT Core Smoke Tests
- User ran `python .\scripts\run_ctat_core_tests.py`.
- All eight CTAT core test functions ran successfully:
  - import/package test
  - decoder token-to-spatial layout test
  - CTABlock residual identity test
  - small CTAT forward shape test
  - modality competition gate test
  - encoder position embedding/gate exposure test
  - alpha scheduler batch-count test
  - HDF5 modality ordering test
- Result: `passed 8`.

### CTAT Feasibility Demo
- User ran `python .\scripts\demo_ctat.py` under `med_ai_310` on CUDA.
- Demo 1 sparsemax vs softmax:
  - Softmax near-zero: `11.0%`, entropy `1.6294`
  - Sparsemax near-zero: `88.2%`, entropy `0.3863`
  - Sparsemax used `13%` as many tokens as softmax.
- Demo 2 alpha-entmax annealing:
  - Active tokens decreased from `9.5/10` at alpha `1.00` to `1.8/10` at alpha `2.00`.
- Demo 3 real-data forward:
  - Loaded images: `(256, 28, 256, 256)`
  - Labels: `(256, 256, 256)`
  - Classes: `102`
  - Model params: `26.9M`
  - Main output: `[2, 102, 256, 256]`
  - Aux outputs: three `[2, 102, 256, 256]`
  - Time: `1.52s` total, `0.759s/img`
- Demo 4 single-subject training:
  - Slices: `128`
  - Initial loss: `4.3675`
  - Final loss: `3.5745`
  - Loss change: `18.2%` decrease
- Script ended with `ALL DEMOS PASSED — CTAT method is feasible`.

### Training Data Check and Single-Subject Script
- User ran `Get-ChildItem -Recurse -Filter *.hdf5`; no output was produced.
- Interpretation: current project has demo/inference assets, but no formal CTAT HDF5 training dataset.
- Added `scripts/train_single_subject_ctat.py` for a tracked single-subject engineering training experiment.
- Script saves:
  - `config.json`
  - `metrics.json`
  - `loss_curve.csv`
  - `final_model.pkl`
- Smoke-tested the script with:
  - `conda run -n med_ai_310 python .\scripts\train_single_subject_ctat.py --iterations 1 --max_slices 4 --exp_dir .\runs\single_subject_script_check --device cuda --log_every 1`
- Smoke-test result:
  - Loaded images `(256, 28, 256, 256)`, labels `(256, 256, 256)`, 102 classes
  - Device: CUDA
  - Slices: 4
  - Params: 26.9M
  - One-step loss: `4.9060`
  - Artifacts saved under `runs/single_subject_script_check`

### Single-Subject CTAT Engineering Experiment
- User ran:
  - `python .\scripts\train_single_subject_ctat.py --iterations 60 --max_slices 128 --exp_dir .\experiments\2026-07-09_single-subject-ctat --device cuda --log_every 5`
- Result:
  - Loaded images `(256, 28, 256, 256)`, labels `(256, 256, 256)`, 102 classes
  - Device: CUDA
  - Slices: `128`
  - Parameters: `26.9M`
  - Initial loss: `4.7899`
  - Final loss: `3.4407`
  - Loss change: `28.2%` decrease
  - Elapsed time from `metrics.json`: `558.99` seconds, about `9.3` minutes
- Verified artifacts under `experiments/2026-07-09_single-subject-ctat`:
  - `config.json`
  - `metrics.json`
  - `loss_curve.csv`
  - `final_model.pkl` (`107,631,505` bytes)
- Interpretation:
  - The CTAT single-subject training path is functional and learns on the bundled real subject.
  - This is engineering validation, not a paper-level result, because it uses one subject and does not test generalization.

### Formal CTAT Training Data Gate
- User requested moving directly to the later/full stage instead of running a 120-iteration repeat.
- Checked `scripts/train_ctat.py` and `data_loader/load_neuroimaging_data.py`.
- Formal CTAT training entrypoint requires a directory containing exactly four modality HDF5 files whose names include:
  - `FractionalAnisotropy`
  - `Trace`
  - `MinEigenvalue`
  - `MidEigenvalue`
- Each HDF5 is expected to contain:
  - `orig_dataset`
  - `aseg_dataset`
  - `weight_dataset`
  - `subject`
- Checked project tree with `Get-ChildItem -Recurse -Filter *.hdf5`; no HDF5 files are currently present.
- Conclusion: full `scripts/train_ctat.py` training cannot start until multi-subject HDF5 data is supplied or generated.

### Demo HDF5 Generation and Formal Entrypoint Smoke
- Added `scripts/create_demo_hdf5_train.py` to generate four modality HDF5 files from `testdata/HCP-100337-b1000`.
- Fixed `AugmentationPadImage` default tuple padding so training transforms define both `pad_size_image` and `pad_size_mask`.
- Updated `scripts/train_ctat.py` to:
  - decode HDF5 subject IDs before subject-level splitting,
  - allow `val_split=0` and single-subject smoke tests,
  - expose `--num_workers`,
  - expose `--max_train_slices` for short smoke tests.
- Generated files under `data/hdf5_train`:
  - `demo_FractionalAnisotropy.hdf5`
  - `demo_Trace.hdf5`
  - `demo_MinEigenvalue.hdf5`
  - `demo_MidEigenvalue.hdf5`
  - `demo_hdf5_manifest.json`
- Verified each HDF5 contains:
  - `orig_dataset`: `(256, 256, 256, 7)`
  - `aseg_dataset`: `(256, 256, 256)`
  - `weight_dataset`: `(256, 256, 256)`
  - `subject`: `(256,)`
- Ran formal HDF5 entrypoint smoke:
  - `conda run -n med_ai_310 python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-formal-smoke --device cuda --val_split 0 --num_workers 0 --max_train_slices 4`
- Smoke result:
  - Loader read all four HDF5 files in modality order.
  - Subject split: `1` train subject, `0` validation subjects.
  - Training used `4` slices.
  - Epoch loss: `3.6508`.
  - Saved `experiments/2026-07-09_ctat-formal-smoke-coronal/final_model.pkl`.

### Coronal HDF5 32-Slice Training
- User ran:
  - `python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-coronal-32slice --device cuda --val_split 0 --num_workers 0 --max_train_slices 32`
- Result:
  - Loader read all four HDF5 modality files.
  - Subject split: `1` train subject, `0` validation subjects.
  - Training used `32` slices.
  - Epoch loss: `3.5304`.
  - Saved `experiments/2026-07-09_ctat-coronal-32slice-coronal/final_model.pkl`.

### Coronal HDF5 Full-Slice 1-Epoch Training
- User ran:
  - `python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-coronal-fullslice --device cuda --val_split 0 --num_workers 0`
- Result:
  - Loader read all four HDF5 modality files.
  - Subject split: `1` train subject, `0` validation subjects.
  - Training used the full demo subject slice set because no `--max_train_slices` cap was set.
  - Epoch loss: `3.8597`.
  - Saved `experiments/2026-07-09_ctat-coronal-fullslice-coronal/final_model.pkl`.
- Verification:
  - Confirmed `final_model.pkl` exists with size `123,019,849` bytes.
  - Confirmed `split_manifest.json` records `n_subjects_total: 1`, `n_train_subjects: 1`, and `val_split: 0.0`.

### Git Upload
- Staged and committed necessary code, workflow memory, and lightweight experiment records.
- Excluded generated data and large artifacts from Git via `.gitignore`:
  - `data/`
  - `runs/`
  - `logs/`
  - `*.hdf5`
  - `*.pkl`
- Verification before upload:
  - `conda run -n med_ai_310 python .\scripts\run_ctat_core_tests.py` -> `passed 8`
  - `conda run -n med_ai_310 python .\scripts\create_demo_hdf5_train.py --help` -> CLI loaded
  - `conda run -n med_ai_310 python .\scripts\train_ctat.py --help` -> CLI loaded with `--num_workers` and `--max_train_slices`
- Commit pushed to GitHub:
  - `299aad6 chore: record CTAT smoke training workflow`
  - Remote: `origin/main`
