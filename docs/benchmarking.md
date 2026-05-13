# Benchmarking

This document covers model benchmarking on recorded CARLA runs. These workflows measure model-forward time, end-to-end runtime, and replayable prediction outputs on saved datasets, without requiring a live CARLA session.

Before using these commands, complete the setup in [installation.md](installation.md). You will need `openpcdet`, `lanedet`, and `carla_app` depending on which benchmark or viewer you want to run.

The benchmarking workflows are also available through the GUI. If you want to run them from the project control interface, see [gui.md](gui.md).

## Contents

* [OpenPCDet Benchmark](#openpcdet-benchmark)
* [LaneDet Benchmark](#lanedet-benchmark)
* [Results Layout](#results-layout)
* [Prediction Viewer](#prediction-viewer)

## OpenPCDet Benchmark

This benchmark loads LiDAR point clouds from a recorded raw CARLA run, runs OpenPCDet frame by frame, measures synchronized model-forward time together with end-to-end per-frame runtime, and can optionally save per-frame predictions for later replay in Rerun.

#### Flags

* `--run-dir` — path to a raw dataset run
* `--cfg-file` — OpenPCDet config file used for inference
* `--ckpt` — checkpoint file to benchmark
* `--score-thresh` — filter low-confidence detections (default: `0.05`)
* `--point-stride` — downsample the point cloud before inference by keeping every Nth point (default: `1`)
* `--warmup` — exclude the first N frames from summary metrics (default: `5`)
* `--save-pred` — save per-frame predictions under the benchmark output directory

#### Examples

```bash
conda activate openpcdet

# Benchmark a recorded raw run with selected config and checkpoint.
python -m tools.benchmark.benchmark_openpcdet --run-dir datasets/raw/{run_XXXX} --cfg-file /path/to/config.yaml --ckpt /path/to/checkpoint.ckpt
```

## LaneDet Benchmark

This benchmark loads front camera frames from a recorded raw CARLA run, runs LaneDet frame by frame, measures synchronized model-forward time together with end-to-end per-frame runtime, and can optionally save both 2D and projected 3D lane predictions for later replay in Rerun.

#### Flags

* `--run-dir` — path to a raw dataset run
* `--config` — LaneDet config file used for inference
* `--ckpt` — checkpoint file to benchmark
* `--score-thresh` — filter low-confidence lane detections (default: `0.2`)
* `--warmup` — exclude the first N frames from summary metrics (default: `5`)
* `--save-pred` — save per-frame predictions under the benchmark output directory

#### Examples

```bash
conda activate lanedet

# Benchmark a recorded raw run with selected config and checkpoint.
python -m tools.benchmark.benchmark_lanedet --run-dir datasets/raw/{run_XXXX} --config /path/to/config.py --ckpt /path/to/checkpoint.pth
```

## Results Layout

Benchmark outputs are organized by raw run name, model family, and timestamp-based run name:

```text
results/
└── benchmark/
    └── run_XXXX/
        ├── openpcdet/
        │   └── {timestamp}/
        │       ├── meta.json
        │       ├── metrics.json
        │       └── predictions/
        │           ├── frame_000000.npz
        │           └── ...
        └── lanedet/
            └── {timestamp}/
                ├── meta.json
                ├── metrics.json
                └── predictions/
                    ├── frame_000000.json
                    └── ...
```

The `metrics.json` file stores summary FPS and timing values, `meta.json` records the benchmark configuration, and `predictions/` is created only when `--save-pred` is enabled.

## Prediction Viewer

This viewer replays a recorded raw CARLA run in Rerun and overlays saved benchmark predictions when object and lane prediction directories are provided.

#### Flags

* `--run-dir` — path to the raw dataset run to replay
* `--objects` — directory with saved OpenPCDet prediction files
* `--lanes` — directory with saved LaneDet prediction files
* `--fps` — replay speed in frames per second (default: `20`)
* `--show-grid` — show the ground grid in the 3D view

#### Examples

```bash
conda activate carla_app

# Replay only OpenPCDet predictions.
python -m tools.benchmark.view_predictions --run-dir datasets/raw/{run_XXXX} --objects results/benchmark/{run_XXXX}/openpcdet/{timestamp}/predictions

# Replay only LaneDet predictions.
python -m tools.benchmark.view_predictions --run-dir datasets/raw/{run_XXXX} --lanes results/benchmark/{run_XXXX}/lanedet/{timestamp}/predictions

# Replay both object and lane predictions together.
python -m tools.benchmark.view_predictions --run-dir datasets/raw/{run_XXXX} --objects results/benchmark/{run_XXXX}/openpcdet/{timestamp}/predictions --lanes results/benchmark/{run_XXXX}/lanedet/{timestamp}/predictions --show-grid
```
