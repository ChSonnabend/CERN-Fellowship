# Softbomb Event Transformer

Transformer framework to classify ALICE events with and without soft bombs. The model consumes per-track quantities and predicts an event label:

- `1`: softbomb sample
- `0`: non-softbomb sample

The default input features are:

- TPC dE/dx
- TOF beta, or `fTOFExpMom` as a documented proxy when no direct beta branch exists
- momentum
- tan(lambda)
- sin(phi)
- event multiplicity

The dataset builder creates a class-balanced event dataset from the two configured Hyperloop/Alien productions.

## Layout

```text
configs/
  softbomb_config.json      Main config for download, dataset creation, training, and submission
  test_config.json          Smaller config for smoke-test dataset/training
  cpu_debug_config.json     CPU-only debug training config for the debug partition
scripts/
  download_alien.py         Download AO2D files with alien_cp
  inspect_root.py           Print available ROOT trees and branches
  build_dataset.py          Create balanced train/validation/test npz datasets
  make_test_dataset.py      Create a small test dataset
  train.py                  Train the transformer
  evaluate.py               Evaluate a trained checkpoint
  export_onnx.py            Export a trained checkpoint to ONNX
  qa_plots.py               Plot confusion matrix and first-layer attention maps
  submit.py                 Generate and submit Slurm jobs from config
  submit_download.sh        Wrapper for download submission
  submit_dataset.sh         Wrapper for full dataset submission
  submit_test_dataset.sh    Wrapper for test dataset submission
  submit_train.sh           Wrapper for GPU training submission
  submit_train_cpu_debug.sh Wrapper for CPU debug training submission
src/softbombs/
  config.py
  root_io.py
  dataset.py
  model.py
  train_utils.py
```

## Quick Start

From this directory:

```bash
python3 scripts/inspect_root.py --config configs/softbomb_config.json --max-files 1
```

If the files are not yet local, download them from Alien:

```bash
python3 scripts/download_alien.py --config configs/softbomb_config.json
```

The downloader only copies files listed in `download.needed_file_patterns` and preserves the path relative to each configured Alien production. For example, an Alien file below `<production>/771/tf1/AO2D.root` is copied to `<local_dir>/771/tf1/AO2D.root`. By default it downloads only `AO2D.root` files. If both `<production>/771/AO2D.root` and `<production>/771/tf1/AO2D.root` exist, it keeps the merged `<production>/771/AO2D.root`; if only the `tf1` file exists, it keeps that.

If the script is already running inside an Alien/O2 environment where `alien_find` or `alien_cp` are available on `PATH`, it calls those commands directly. Otherwise it wraps them with `alienv enter O2/latest -- ...`.

Wildcard components in `input.classes[].alien_path` are supported. For example, `<production>/2*` is expanded with `alien_ls <production>`, then `alien_find` is run on each concrete matching directory. The local path remains relative to the non-wildcard production base, so `<production>/201/tf1/AO2D.root` becomes `<local_dir>/201/tf1/AO2D.root`.

For a dry-run file list:

```bash
python3 scripts/download_alien.py --config configs/softbomb_config.json --dry-run
```

Build the balanced dataset:

```bash
python3 scripts/build_dataset.py --config configs/softbomb_config.json
```

Settings such as `dataset.max_tracks`, `dataset.include_event_multiplicity`, feature aliases, track sorting, and split fractions are baked into the generated `.npz` files. If you change them, rebuild the dataset before training; otherwise training will continue to read the old array shape. The training script checks `metadata.json` and stops with a clear error if the dataset no longer matches the config.

For the transformer, `dataset.max_tracks` controls the token sequence length and attention memory scales roughly as `batch_size * n_heads * max_tracks^2`. Values like `16384` are not practical for the current vanilla transformer on a 32 GB MI100. The default `2048` covers more than 99.5% of the current training events while keeping memory manageable.

Train locally or inside an interactive GPU allocation:

```bash
python3 scripts/train.py --config configs/softbomb_config.json
```

Evaluate the best checkpoint:

```bash
python3 scripts/evaluate.py --config configs/softbomb_config.json
```

Export the best checkpoint to ONNX:

```bash
python3 scripts/export_onnx.py --config configs/softbomb_config.json
```

Training jobs export `best_model.onnx` automatically when `training.export_onnx` is `true`.

Create QA plots:

```bash
python3 scripts/qa_plots.py --config configs/softbomb_config.json --split holdout
```

## Slurm Submission

Everything is steered through `configs/softbomb_config.json`.

Submit the download job:

```bash
bash scripts/submit_download.sh configs/softbomb_config.json
```

Submit full dataset creation:

```bash
bash scripts/submit_dataset.sh configs/softbomb_config.json
```

Submit a small test dataset job:

```bash
bash scripts/submit_test_dataset.sh configs/test_config.json
```

Submit transformer training on the GPU partition:

```bash
bash scripts/submit_train.sh configs/softbomb_config.json
```

Submit standalone ONNX export:

```bash
bash scripts/submit_export_onnx.sh configs/softbomb_config.json
```

The submission script writes generated job files to `jobs/`. By default Slurm stdout/stderr are sent to `/dev/null` first and then redirected into the job output directory, so each submitted job has a single useful log location. Set `slurm.keep_central_slurm_logs` to `true` only if you also want bootstrap logs in the central `outputs/.../slurm/` directory.

Training jobs submitted through Slurm write checkpoints, history, QA, and ONNX files to a unique job directory:

```text
outputs/softbomb_transformer/jobs/YYYYMMDD_SLURMJOBID/training/
outputs/softbomb_transformer/jobs/YYYYMMDD_SLURMJOBID/slurm/
```

This lets multiple training jobs run in parallel without overwriting `best_model.pt`, `history.csv`, `best_model.onnx`, or the main stdout/stderr logs. The static `training.output_dir` in the JSON config is still used for local/manual runs unless `SOFTBOMB_TRAINING_OUTPUT_DIR` is set.

Training jobs also run holdout evaluation and QA by default when `training.evaluate_after_train` and `training.qa_plots_after_train` are true. The job-specific `training/` directory will contain:

```text
holdout_metrics.json
holdout_confusion_matrix.csv
holdout_predictions.csv
qa/holdout_confusion_matrix.pdf
qa/holdout_event0_first_layer_attention.pdf
```

Python stages submitted through `scripts/submit.py` run inside the configured Apptainer image:

```text
/lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/cuda_torch_env.sif
```

For GPU jobs the generated command is `apptainer exec --nv ... python3 ...`.

The Slurm `device` setting can be switched between NVIDIA and AMD:

```json
"device": "NVIDIA_H200_GPU"
```

uses `nvidia_gpu`, `h200`, `cuda_torch_env.sif`, and `apptainer exec --nv`.

```json
"device": "AMD_MI100_GPU"
```

uses `amd_gpu`, `mi100`, `rocm_torch_env.sif`, and plain `apptainer exec`.
An AMD-ready config is provided as `configs/amd_config.json`, and training can be submitted with:

```bash
bash scripts/submit_train_amd.sh
```

An EPN-ready config following the `o2-tpc-pid` convention is provided as `configs/epn_config.json`. It submits to `prod`, requests GPUs with `--gres`, runs through `srun`, and uses direct `python3.9` after `module load O2PDPSuite`. No Apptainer container is used for EPN jobs:

```bash
bash scripts/submit_train_epn.sh
```

For a pure CPU debug submission, use:

```bash
bash scripts/submit_train_cpu_debug.sh
```

This uses `configs/cpu_debug_config.json`, submits to the `debug` partition, requests no GPU, sets `training.device` to `cpu`, and uses the CUDA Apptainer image only as the Python environment.

Generated Apptainer jobs create a per-job runtime directory below `slurm.cache_dir` and route Apptainer, Torch, Triton, MIOPEN, ROCm, AMD COMGR, and Python bytecode caches there. The Python `TMPDIR` is kept short below `/tmp` so PyTorch DataLoader multiprocessing does not hit Unix socket path-length limits.

The generated job scripts install an `EXIT`/`INT`/`TERM` trap that removes only the guarded per-job runtime/cache directory and short `/tmp` directory when the job finishes or is cancelled. The job output directory with logs, checkpoints, QA, and ONNX files is kept.

## Config Notes

The important knobs are in `configs/softbomb_config.json`:

- `input.classes`: labeled signal/background Alien paths and local raw directories.
- `dataset.tree_name`: set this if auto-detection finds the wrong ROOT tree.
- `dataset.features`: branch aliases used to find TPC dE/dx, TOF beta, momentum, tan(lambda), and sin(phi).
- `dataset.event_id_branches`: branch aliases used to group tracks into events.
- `dataset.max_tracks`: tracks per event after truncation/padding.
- `dataset.max_events_per_class`: optional cap after class balancing.
- `training.model`: transformer depth, width, attention heads, dropout.
- `slurm`: GPU partition, time, memory, container, and setup commands.

For the downloaded AO2Ds in this directory, the builder joins `O2track_iu` and `O2trackextra_002` by row index:

- `O2track_iu`: event id, `tan(lambda)`, `sin(phi)`, signed inverse pT
- `O2trackextra_002`: TPC dE/dx and available TOF PID quantities

The sample AO2Ds do not contain a direct `fTOFBeta` branch. The framework therefore records `tof_pid_proxy` in `metadata.json` and uses `fTOFExpMom` by default. If a later AO2D contains direct beta, add that branch to `dataset.features.tof_beta.branches` and it will be preferred automatically.

If auto-detection fails, run `inspect_root.py` and update `dataset.track_tree_name`, `dataset.track_extra_tree_name`, `dataset.event_id_branches`, or the feature aliases in the config.

## O2 / Alien Environment

The download script runs Alien commands through:

```bash
alienv enter O2/latest -- alien_find ...
alienv enter O2/latest -- alien_cp ...
```

with `download.o2_workdir` set by default to:

```text
/scratch/alice/csonnab/MyO2
```

The downloaded files are stored under `data/raw/softbomb` and `data/raw/background` unless changed in the config.

## Outputs

Dataset files:

```text
outputs/softbomb_transformer/data/train.npz
outputs/softbomb_transformer/data/val.npz
outputs/softbomb_transformer/data/test.npz
outputs/softbomb_transformer/data/metadata.json
outputs/softbomb_transformer/data/scaler.npz
```

Training files:

```text
outputs/softbomb_transformer/training/best_model.pt
outputs/softbomb_transformer/training/best_model.onnx
outputs/softbomb_transformer/training/best_model.onnx.json
outputs/softbomb_transformer/training/last_model.pt
outputs/softbomb_transformer/training/history.csv
outputs/softbomb_transformer/training/test_metrics.json
outputs/softbomb_transformer/training/test_predictions.csv
```
