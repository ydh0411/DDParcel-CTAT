# Task Plan: DDParcel-CTAT Experiments

## Goal
Set up DDParcel-CTAT assets and run experiments step by step, starting from the smallest reproducible public demo before moving to CTAT training or full-paper datasets.

## Current Phase
Phase 1: download and verify public release assets.

## Phases

### Phase 1: Public Demo Assets
- [x] Download or transfer `weights/` from the original DDParcel release.
- [x] Download or transfer `testdata/` from the original DDParcel release.
- [x] Download `100HCP-population-mean-T2-1mm.nii.gz`.
- [x] Verify asset presence and counts.
- [x] Confirm release assets are available as folders in the project root.
- **Status:** complete

### Phase 2: Environment Verification
- [x] Identify a Python environment with torch.
- [x] Check project runtime imports under that environment.
- [x] Decide whether to install missing lightweight tools such as `pytest`.
- **Status:** complete

### Phase 3: Baseline DDParcel Demo
- [x] Run DDParcel baseline inference on the provided preprocessed test data.
- [x] Record command, output path, runtime, and any errors.
- [x] Verify output MGZ is readable and has expected 256³ shape.
- [x] Compare reproduced output against bundled reference output.
- **Status:** complete

### Phase 4: CTAT Smoke Tests
- [ ] Run CTAT core tests under `med_ai_310`.
- [ ] Run `scripts/demo_ctat.py` if required inputs are present.
- [ ] Run one CTAT inference smoke test if a compatible checkpoint exists.
- [x] Run CTAT core tests under `med_ai_310`.
- [x] Run `scripts/demo_ctat.py` on the bundled HCP subject.
- [ ] Run one CTAT inference smoke test if a compatible checkpoint exists.
- **Status:** mostly complete; CTAT checkpoint inference is pending because no trained CTAT checkpoint is present.

### Phase 5: Training Data Plan
- [x] Separate public demo assets from full training datasets.
- [x] Confirm no `.hdf5` training files are present in the current project tree.
- [ ] List HCP/CNP/PPMI access steps and required accounts.
- [ ] Prepare subject-level split and experiment directory only after data access is settled.
- [x] Add a single-subject CTAT training script for engineering validation without `.hdf5`.
- [x] Generate demo-subject HDF5 files for formal training-entry smoke tests.
- **Status:** in_progress

### Phase 6: Single-Subject CTAT Engineering Experiment
- [x] Add reproducible script that saves config, metrics, loss curve, and checkpoint.
- [x] Smoke-test the script with 1 iteration and 4 slices.
- [x] Run the full single-subject engineering experiment.
- [x] Review saved metrics and decide whether to proceed to full training data preparation.
- **Status:** complete; the next research step is obtaining or building subject-level multi-subject training HDF5 data.

### Phase 7: Formal HDF5 Entrypoint Smoke Test
- [x] Create four HDF5 files under `data/hdf5_train`.
- [x] Verify HDF5 datasets and shapes.
- [x] Run `scripts/train_ctat.py` on the generated HDF5 files with a 4-slice cap.
- [x] Verify checkpoint and split manifest artifacts.
- **Status:** complete for smoke testing; not paper evidence.

### Phase 8: Single-Subject HDF5 Coronal Training
- [x] Run 32-slice capped coronal HDF5 training.
- [x] Run full 256-slice coronal HDF5 training for one epoch.
- [ ] Run longer coronal training if useful for engineering stability.
- [ ] Prepare true multi-subject HDF5 data before paper-level claims.
- **Status:** in_progress

## Decisions
- Use `med_ai_310` for torch-based checks because it has `torch 2.10.0+cu130` and CUDA available.
- Start with DDParcel release demo assets because full HCP/CNP/PPMI data requires external access agreements/accounts.
- Do not treat the demo `testdata.zip` as the full training dataset.

## Download URLs
- Weights: `https://github.com/zhangfanmark/DDParcel/releases/download/pre-release/weights.zip`
- Demo data: `https://github.com/zhangfanmark/DDParcel/releases/download/pre-release/testdata.zip`
- HCP T2 template: `https://github.com/zhangfanmark/DDParcel/releases/download/pre-release/100HCP-population-mean-T2-1mm.nii.gz`

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Base Python lacks `torch` | `python -c "import torch"` | Use `conda run -n med_ai_310 python ...`. |
| `med_ai_310` lacks `pytest` | `conda run -n med_ai_310 python -m pytest` | Directly ran test functions for initial verification; install pytest later if needed. |
