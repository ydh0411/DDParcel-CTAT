# CTAT Coronal 128-Slice 3-Epoch HDF5 Training

## Question
Can the formal HDF5 training entrypoint complete a longer, bounded coronal training run without the full 10-epoch run stalling?

## Change From Baseline
This run uses the generated demo-subject HDF5 files, caps training to 128 slices, and trains the coronal CTAT model for 3 epochs.

## Config
- Command: `python .\scripts\train_ctat.py --hdf5_dir .\data\hdf5_train --view coronal --num_classes 102 --embed_dim 48 --epochs 3 --batch_size 1 --exp_dir .\experiments\2026-07-10_ctat-coronal-128slice-3epoch --device cuda --val_split 0 --num_workers 0 --max_train_slices 128`
- Data: `data/hdf5_train`
- View: `coronal`
- Classes: `102`
- Epochs: `3`
- Train slices used: `128`
- Validation split: `0`
- Device: CUDA

## Metrics
- Epoch 1 loss: `3.9417`
- Epoch 2 loss: `3.6869`
- Epoch 3 loss: `3.5931`
- Final checkpoint: `final_model.pkl`

## Supported Conclusion
The formal HDF5 training path can complete a bounded multi-epoch coronal run, and the training loss decreased over all three epochs.

## Unsupported Conclusion
This run does not demonstrate generalization because it uses one demo subject and no validation split.

## Next Experiment
Run a full 256-slice 3-epoch coronal experiment, or prepare true multi-subject HDF5 data for validation.
