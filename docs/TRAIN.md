# Training

This document covers dataset preparation, training, evaluation, and result layouts for the two model stacks used in the project: OpenPCDet and LaneDet.

Before using these workflows, complete the setup in [Installation](INSTALL.md). In particular, make sure the required Conda environment and CUDA toolchain are ready for the model family you want to use.

All workflows described here are also available through the GUI. If you want to use the graphical workflow layer for preparation, training, and evaluation, continue with [GUI](GUI.md).

## Contents

* [OpenPCDet](#openpcdet)
* [LaneDet](#lanedet)

## OpenPCDet

The OpenPCDet framework is used here as a bundled Git submodule under `third_party/OpenPCDet`.

The integration is based on the upstream [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) project, with this repository currently pointing to a project-specific fork: [Camill0x/OpenPCDet](https://github.com/Camill0x/OpenPCDet).

Use the `openpcdet` environment for all OpenPCDet preparation, training, evaluation, and live inference commands:

```bash
conda activate openpcdet
```

### Class family

The OpenPCDet workflow in this project uses a six-class CARLA/nuScenes-style class family (`carla_nuscenes6`):

* `car`
* `truck`
* `bus`
* `motorcycle`
* `bicycle`
* `pedestrian`

The training and fine-tuning workflow is built around adapting nuScenes-oriented OpenPCDet baselines to this reduced class set. The relevant upstream model baselines are documented here:

* [OpenPCDet nuScenes 3D object detection baselines](https://github.com/open-mmlab/OpenPCDet?tab=readme-ov-file#nuscenes-3d-object-detection-baselines)

### Project-specific additions in the fork

Compared with upstream OpenPCDet, this project adds a CARLA-specific dataset and config family inside the submodule under `third_party/OpenPCDet/`.

The main project-added files are:

* `pcdet/datasets/carla_nuscenes6/carla_nuscenes6_dataset.py`
* `pcdet/datasets/carla_nuscenes6/carla_nuscenes6_eval.py`
* `tools/cfgs/carla_nuscenes6_models/carla_nuscenes6_centerpoint_pp_zeroshot.yaml`
* `tools/cfgs/carla_nuscenes6_models/carla_nuscenes6_centerpoint_pp_ft.yaml`
* `tools/cfgs/carla_nuscenes6_models/carla_nuscenes6_transfusion_zeroshot.yaml`
* `tools/cfgs/carla_nuscenes6_models/carla_nuscenes6_transfusion_ft.yaml`
* `tools/cfgs/dataset_configs/carla_nuscenes6_centerpoint_dataset.yaml`
* `tools/cfgs/dataset_configs/carla_nuscenes6_transfusion_dataset.yaml`

Together, these files cover the project dataset integration, evaluation logic, and the config set used by the current OpenPCDet presets.

### Dataset preparation

This tool builds a prepared OpenPCDet dataset view from recorded raw CARLA runs.

It does not duplicate raw point clouds. Instead, it:
* Selects one or more `run_XXXX` directories under `datasets/raw`
* Scans all `frame_*` directories
* Builds sample metadata and split files
* Writes an OpenPCDet-ready dataset root under `datasets/openpcdet/<class_filter>/<name>`

#### Flags

* `--class-filter` — select the class family and config family (default: `carla_nuscenes6`)
* `--name` — choose the prepared dataset variant name (default: `default`)
* `--all` — use every `run_XXXX` directory under `datasets/raw`
* `--runs` — select only specific raw runs
* `--val-ratio` — set the validation split ratio (default: `0.2`)
* `--test-ratio` — set the test split ratio (default: `0.2`)
* `--seed` — control the split shuffle seed (default: `42`)

#### Examples

```bash
conda activate openpcdet

# Build default dataset from all collected raw runs.
python -m tools.openpcdet.prepare_dataset --all

# Build dataset from selected runs only.
python -m tools.openpcdet.prepare_dataset --runs run_0001 run_0002

# Build dataset with custom split ratio.
python -m tools.openpcdet.prepare_dataset \
  --name split_01 \
  --all \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 7
```

#### Default output layout

```text
datasets/
└── openpcdet/
    └── carla_nuscenes6/
        └── default/
            ├── ImageSets/
            │   ├── train.txt
            │   ├── val.txt
            │   └── test.txt
            └── infos/
                ├── infos_train.pkl
                ├── infos_val.pkl
                ├── infos_trainval.pkl
                └── infos_test.pkl
```

### Training

The training workflow in this repository is implemented as a project-owned wrapper around the upstream OpenPCDet training entrypoint in `third_party/OpenPCDet/tools/train.py`.

After the upstream run finishes, the wrapper reorganizes checkpoints, logs, config snapshots, and metadata into the project-owned `results/openpcdet/train/` layout.

Internally, it resolves either a project preset or an explicit OpenPCDet config, creates a temporary work directory, builds the argument list for the upstream `train.py`, and injects the prepared dataset path through config overrides.

After training completes, it gathers the produced checkpoints, runs validation-based checkpoint selection, keeps `best.ckpt` and `last.ckpt`, and copies logs, TensorBoard files, config snapshots, and compact metadata into the final run directory.

#### Presets

The current `--preset` options are:

* `cn6-transfusion-ft` — fine-tuning preset for the `carla_nuscenes6` TransFusion configuration
* `cn6-transfusion-zeroshot` — preset for directly testing a nuScenes-pretrained TransFusion checkpoint on the `carla_nuscenes6` dataset before fine-tuning
* `cn6-centerpoint-pp-ft` — fine-tuning preset for the `carla_nuscenes6` CenterPoint-PointPillar configuration
* `cn6-centerpoint-pp-zeroshot` — preset for directly testing a nuScenes-pretrained CenterPoint-PointPillar checkpoint on the `carla_nuscenes6` dataset before fine-tuning

Each preset resolves to a project config under:

* `third_party/OpenPCDet/tools/cfgs/carla_nuscenes6_models/`

#### Flags

* `--dataset-name` — select the prepared dataset variant (default: `default`)
* `--preset` — choose one of the project-managed presets
* `--cfg-file` — use an explicit OpenPCDet config instead of `--preset`
* `--pretrained-model` — initialize from pretrained model
* `--ckpt` — resume from an existing checkpoint
* `--batch-size` — override batch size
* `--epochs` — override epoch count
* `--workers` — control dataloader workers (default: `4`)
* `--keep-all-ckpt` — keep per-epoch checkpoints
* `--best-metric` — select the validation metric (default: `mAP`)
* `--set ...` — pass extra OpenPCDet config overrides

#### Pretrained checkpoints

Training can start from pretrained weights through `--pretrained-model`.

In this project, the fine-tuning presets were designed around the upstream nuScenes checkpoints:

* `cn6-transfusion-ft` — use the upstream `TransFusion-L` checkpoint
* `cn6-centerpoint-pp-ft` — use the upstream `CenterPoint-PointPillar` checkpoint

Pretrained checkpoints can be downloaded from:

* [OpenPCDet nuScenes 3D object detection baselines](https://github.com/open-mmlab/OpenPCDet?tab=readme-ov-file#nuscenes-3d-object-detection-baselines)

#### Examples

```bash
conda activate openpcdet

# Train from a project preset using pretrained model.
python -m tools.openpcdet.train \
    --preset cn6-transfusion-ft \
    --pretrained-model /path/to/pretrained_transfusion_model.pth

# Resume training from a specified dataset name, checkpoint and config.
python -m tools.openpcdet.train \
  --dataset-name {name} \
  --cfg-file /path/to/config.yaml \
  --ckpt /path/to/checkpoint.ckpt
```

### Evaluation

The evaluation workflow in this repository is implemented as a project-owned wrapper around the upstream OpenPCDet evaluation entrypoint in `third_party/OpenPCDet/tools/test.py`.

It evaluates a checkpoint on the held-out test split and saves the resulting artifacts under `results/openpcdet/test/`.

The wrapper resolves the preset or explicit config, prepares a clean evaluation work directory, builds the upstream `test.py` command with the prepared dataset and test split paths, runs the evaluation, and then copies the result files and compact metadata into the final output directory.

#### Flags

* `--dataset-name` — select the prepared dataset variant (default: `default`)
* `--preset` — choose a project-managed config preset
* `--cfg-file` — use an explicit OpenPCDet config instead of `--preset`
* `--ckpt` — point to the checkpoint to evaluate
* `--batch-size` — override evaluation batch size
* `--workers` — control dataloader workers (default: `4`)
* `--set ...` — pass extra OpenPCDet config overrides

#### Examples

```bash
conda activate openpcdet

# Evaluate pretrained Transfusion checkpoint.
python -m tools.openpcdet.test \
    --preset cn6-transfusion-zeroshot \
    --ckpt /path/to/pretrained_transfusion_ckpt.pth

# Evaluate checkpoint from a Transfusion model training.
python -m tools.openpcdet.test \
  --preset cn6-transfusion-ft \
  --ckpt results/openpcdet/train/carla_nuscenes6/{timestamp}/ckpt/best.ckpt

# Evaluate with an explicit dataset name, checkpoint and config.
python -m tools.openpcdet.test \
    --dataset-name {name} \
    --cfg-file /path/to/config.yaml \
    --ckpt /path/to/checkpoint.ckpt
```

### Results layout

OpenPCDet outputs are organized by mode, class filter, and timestamp-based run name:

```text
results/
└── openpcdet/
    ├── train/
    │   └── carla_nuscenes6/
    │       └── {timestamp}/
    │           ├── ckpt/
    │           │   ├── best.ckpt
    │           │   ├── last.ckpt
    │           │   └── epochs/            optional if --keep-all-ckpt
    │           ├── meta.json
    │           ├── metrics.json
    │           ├── source_config.yaml
    │           ├── tensorboard/
    │           └── train.log
    └── test/
        └── carla_nuscenes6/
            └── {timestamp}/
                ├── meta.json
                ├── metrics.json
                ├── result.pkl
                ├── source_config.yaml
                └── test.log
```

## LaneDet

The LaneDet framework is used here as a bundled Git submodule under `third_party/lanedet`.

The integration is based on the upstream [LaneDet](https://github.com/Turoad/lanedet) project, with this repository currently pointing to a project-specific fork: [Camill0x/lanedet](https://github.com/Camill0x/lanedet)

Use the `lanedet` environment for all LaneDet preparation, training, evaluation, and live inference commands:

```bash
conda activate lanedet
```

### Dataset preparation

This tool builds a prepared LaneDet dataset from recorded raw CARLA runs. The current implementation supports TuSimple dataset format only.

It selects one or more `run_XXXX` directories under `datasets/raw`, scans the recorded frames, loads lane annotations, filters unusable samples, generates train/val/test splits, and writes RGB clips, segmentation labels, and TuSimple-style JSON annotations into the final dataset root under `datasets/lanedet/<format>/<name>`.

#### Flags

* `--format` — select the prepared LaneDet dataset family (default: `tusimple`)
* `--name` — choose the prepared dataset variant name (default: `default`)
* `--all` — use every `run_XXXX` directory under `datasets/raw`
* `--runs` — select only specific raw runs
* `--max-lanes` — limit how many lanes are retained per sample (default: `5`)
* `--line-width` — control segmentation mask line width in pixels (default: `10`)
* `--val-ratio` — set the validation split ratio (default: `0.2`)
* `--test-ratio` — set the test split ratio (default: `0.2`)
* `--seed` — control the split shuffle seed (default: `42`)

#### Examples

```bash
conda activate lanedet

# Build the default TuSimple-like prepared dataset from all raw runs.
python -m tools.lanedet.prepare_dataset --format tusimple --all

# Build dataset from selected runs only.
python -m tools.lanedet.prepare_dataset --format tusimple --runs run_0001 run_0002

# Build a dataset with a custom lane cap and line width.
python -m tools.lanedet.prepare_dataset \
  --format tusimple \
  --all \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --name split_01
```

#### Output layout

```text
datasets/
└── lanedet/
    └── tusimple/
        └── default/
            ├── clips/
            │   ├── run_XXXX/
            │   │   ├── frame_XXXXXX.png
            │   │   └── ...
            │   └── ...
            ├── seg_label/
            │   ├── run_XXXX/
            │   │   ├── frame_XXXXXX.png
            │   │   └── ...
            │   └── ...
            ├── label_data_0313.json
            ├── label_data_0601.json
            ├── label_data_0531.json
            └── test_label.json
```

The annotation filenames follow the TuSimple naming convention used by the original LaneDet tooling. In this project:
* `label_data_0313.json`, `label_data_0601.json` — together cover the training split
* `label_data_0531.json` — covers validation split
* `test_label.json` — covers test split

### Training

The training workflow in this repository is implemented as a project-owned wrapper around the upstream LaneDet entrypoint in `third_party/lanedet/main.py`.

It launches training from the main project repo and then post-processes the resulting artifacts into the project-owned `results/lanedet/train/` layout.

The wrapper resolves either a preset or an explicit config, generates a runtime config adjusted to the selected dataset and overrides, launches the upstream LaneDet training command, and then collects logs, predictions, metrics, checkpoints, and metadata into the final run directory.

#### Presets

The current `--preset` options are:

* `laneatt-r34-tusimple` — project preset for the LaneATT ResNet-34 configuration on the TuSimple-like dataset

Each preset resolves to a project config under:

* `third_party/lanedet/configs/`

#### Flags

* `--preset` — choose a project-managed config. preset
* `--config` — use an explicit LaneDet config instead of a preset
* `--data-root` — override the dataset path from the config
* `--batch-size` — override batch size
* `--epochs` — override epoch count
* `--workers` — override dataloader workers
* `--finetune-from` — initialize from existing weights
* `--gpus` — select GPU ids (default: `[0]`)
* `--seed` — set the random seed (default: `0`)
* `--view` — enable LaneDet visualization mode

When `--preset` is used without `--data-root`, the workflow defaults to the prepared dataset under `datasets/lanedet/tusimple/default`. Use `--data-root` to point the preset at a different prepared dataset variant.

#### Pretrained checkpoints

Training can start from pretrained weights through `--finetune-from`.

In this project, the `laneatt-r34-tusimple` preset is paired with the upstream `laneatt_r34_tusimple.zip` checkpoint package.

Pretrained checkpoints can be downloaded from:

* [LaneDet v1.0 release page](https://github.com/Turoad/lanedet/releases/tag/1.0)

#### Examples

```bash
conda activate lanedet

# Fine-tune from pretrained weights.
python -m tools.lanedet.main \
    --preset laneatt-r34-tusimple \
    --finetune-from /path/to/laneatt_r34_tusimple.pth

# Train from the project preset and specified data root.
python -m tools.lanedet.main \
    --preset laneatt-r34-tusimple \
    --data-root datasets/lanedet/tusimple/{name}

# Train from an explicit LaneDet config.
python -m tools.lanedet.main \
    --config /path/to/config.py \
    --data-root dataset/lanedet/tusimple/{name}
```

### Evaluation

The same project-owned LaneDet entrypoint switches into validation mode when `--validate` is provided.

In validation mode, the wrapper resolves the preset or explicit config, requires an evaluation checkpoint through `--load-from`, generates a runtime config for the selected dataset, runs the upstream LaneDet validation command, and then collects logs, predictions, metrics, and metadata into the final run directory.

#### Flags

* `--preset` — choose a project-managed config preset
* `--config` — use an explicit LaneDet config instead of a preset
* `--data-root` — override the dataset path from the config
* `--validate` — switch from training mode to validation mode
* `--load-from` — required in validation mode; point to the checkpoint
* `--gpus` — select GPU ids (default: `[0]`)
* `--view` — enable LaneDet visualization mode

#### Examples

```bash
conda activate lanedet

# Evaluate a pretrained LaneATT ResNet-34 TuSimple checkpoint.
python -m tools.lanedet.main \
  --preset laneatt-r34-tusimple \
  --validate \
  --load-from /path/to/laneatt_r34_tusimple.pth \
  --view

# Evaluate checkpoint from a previous LaneATT TuSimple training run.
python -m tools.lanedet.main \
  --config results/lanedet/train/laneatt/tusimple/{timestamp}/config.py \
  --validate \
  --load-from results/lanedet/train/laneatt/tusimple/{timestamp}/ckpt/best.pth \
  --view

# Evaluate with an explicit data root and checkpoint.
python -m tools.lanedet.main \
  --preset laneatt-r34-tusimple \
  --data-root datasets/lanedet/tusimple/{name} \
  --validate \
  --load-from /path/to/checkpoint.pth \
  --view
```

### Results layout

LaneDet outputs are organized by mode, model family, dataset family, and timestamp-based run name:

```text
results/
└── lanedet/
    ├── train/
    │   └── laneatt/
    │       └── tusimple/
    │           └── {timestamp}/
    │               ├── ckpt/
    │               │   ├── best.pth
    │               │   └── last.pth
    │               ├── config.py
    │               ├── meta.json
    │               ├── metrics.json
    │               ├── predictions.json
    │               ├── train.log
    │               └── visualization/     optional if --view
    └── test/
        └── laneatt/
            └── tusimple/
                └── {timestamp}/
                    ├── config.py
                    ├── meta.json
                    ├── metrics.json
                    ├── predictions.json
                    ├── test.log
                    └── visualization/     optional if --view
```
