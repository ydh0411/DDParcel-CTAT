# Single-Subject CTAT Engineering Experiment

## Question
Can the CTAT training path run end to end on the bundled real HCP demo subject, including CUDA execution, loss computation, optimization, logging, and checkpoint saving?

## Change From Baseline
This is not a DDParcel baseline inference run. It trains the CTAT model on one bundled preprocessed subject for a small engineering validation.

## Config
- Command: `python .\scripts\train_single_subject_ctat.py --iterations 60 --max_slices 128 --exp_dir .\experiments\2026-07-09_single-subject-ctat --device cuda --log_every 5`
- Data: `testdata/HCP-100337-b1000`
- Device: CUDA
- Seed: `42`
- Iterations: `60`
- Slices: `128`
- Classes: `102`
- Model parameters: `26,873,753`
- Checkpoint: `final_model.pkl`

## Metrics
- Initial loss: `4.7899`
- Final loss: `3.4407`
- Loss change: `28.2%` decrease
- Elapsed time: `558.99` seconds, about `9.3` minutes

## Supported Conclusion
The CTAT model, loss, optimizer, alpha schedule, CUDA path, logging, and checkpoint saving work on real bundled DDParcel demo data.

## Unsupported Conclusion
This experiment does not show segmentation accuracy, robustness, or generalization. It uses one subject and should not be used as paper-level evidence.

## Next Experiment
Either run a slightly longer engineering stability repeat, or move to multi-subject HDF5 data preparation before claiming model performance.
