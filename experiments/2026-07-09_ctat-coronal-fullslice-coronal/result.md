# CTAT Coronal Full-Slice HDF5 Training

## Question
Can the formal HDF5 training entrypoint train for one epoch over the full 256-slice demo subject?

## Change From Baseline
This run removes the `--max_train_slices` cap and trains the coronal CTAT model on all slices from the generated single-subject HDF5 files.

## Config
- Command: `python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-coronal-fullslice --device cuda --val_split 0 --num_workers 0`
- Data: `data/hdf5_train`
- View: `coronal`
- Classes: `102`
- Epochs: `1`
- Train slices used: `256`
- Validation split: `0`
- Device: CUDA

## Metrics
- Epoch loss: `3.8597`
- Final checkpoint: `final_model.pkl`

## Supported Conclusion
The formal HDF5 training path can complete a full single-subject coronal epoch and save a checkpoint.

## Unsupported Conclusion
This run does not measure held-out performance or generalization because it uses one demo subject and no validation split.

## Next Experiment
Run a longer coronal training job or prepare true multi-subject HDF5 data for paper-level experiments.
