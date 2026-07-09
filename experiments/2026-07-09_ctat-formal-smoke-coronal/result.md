# CTAT Formal HDF5 Entry Smoke Test

## Question
Can the formal `scripts/train_ctat.py` entrypoint load HDF5 training files and run CTAT training through `CTATSolver`?

## Change From Baseline
The demo NIfTI subject was converted into four modality HDF5 files under `data/hdf5_train`. This smoke test used the formal HDF5 loader instead of the single-subject engineering script.

## Config
- Command: `conda run -n med_ai_310 python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-formal-smoke --device cuda --val_split 0 --num_workers 0 --max_train_slices 4`
- Data: `data/hdf5_train`
- View: `coronal`
- Classes: `102`
- Epochs: `1`
- Train slices used: `4`
- Validation split: `0`
- Device: CUDA

## Metrics
- Epoch loss: `3.6508`
- Final checkpoint: `final_model.pkl`

## Supported Conclusion
The formal CTAT HDF5 training entrypoint can load the generated HDF5 files, build a subject-level split, train for one epoch, and save a checkpoint.

## Unsupported Conclusion
This does not demonstrate accuracy or generalization. It uses one demo subject and only four training slices.

## Next Experiment
Run full single-subject HDF5 coronal training without `--max_train_slices`, or prepare real multi-subject HDF5 data before paper-level experiments.
