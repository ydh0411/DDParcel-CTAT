# CTAT Coronal 32-Slice HDF5 Training

## Question
Can the formal HDF5 training entrypoint run beyond the minimal 4-slice smoke test while still keeping runtime short?

## Change From Baseline
This run uses the generated demo-subject HDF5 files and trains the coronal CTAT model for one epoch on 32 capped slices.

## Config
- Command: `python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 1 --batch_size 1 --exp_dir .\experiments\2026-07-09_ctat-coronal-32slice --device cuda --val_split 0 --num_workers 0 --max_train_slices 32`
- Data: `data/hdf5_train`
- View: `coronal`
- Classes: `102`
- Epochs: `1`
- Train slices used: `32`
- Validation split: `0`
- Device: CUDA

## Metrics
- Epoch loss: `3.5304`
- Final checkpoint: `final_model.pkl`

## Supported Conclusion
The formal HDF5 training path runs on a larger capped subset than the initial smoke test and saves a checkpoint.

## Unsupported Conclusion
This run does not measure accuracy or generalization. It uses one demo subject and 32 slices.

## Next Experiment
Run the full 256-slice single-subject coronal HDF5 training.
