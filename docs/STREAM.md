# Streaming

This document covers the live pipeline used for real-time scene reconstruction from CARLA. These workflows publish synchronized sensor frames, run detector nodes in their own environments, merge the predictions into a single scene stream, and render the result in Rerun.

Before using these commands, complete the setup in [Installation](INSTALL.md). You will need the `carla_app`, `openpcdet`, and `lanedet` environments depending on which pipeline nodes you want to run.

For simulator driving, see [CARLA](CARLA.md).

These workflows are also available through the GUI. If you want to operate the pipeline from the project control interface, see [GUI](GUI.md).


## Contents

* [Pipeline Overview](#pipeline-overview)
* [Producer](#producer)
* [Aggregator](#aggregator)
* [Visualizer](#visualizer)
* [OpenPCDet Inference Node](#openpcdet-inference-node)
* [LaneDet Inference Node](#lanedet-inference-node)

## Pipeline Overview

The live pipeline is split into small cooperating processes:

* Producer reads CARLA sensors and publishes synchronized frames into shared memory
* Detector nodes read those frames and write predictions back into shared memory
* Aggregator combines state, object predictions, and lane predictions into one scene stream
* Visualizer consumes that stream and renders it in Rerun

The detector nodes can be attached or detached at any time while the producer, aggregator, and visualizer keep running.

For a full live run, the practical order is:

1. Start the simulator with `./carla_server.sh`

2. Start manual control in synchronous mode with `conda activate carla_app && python -m tools.carla.manual_control --sync`

3. Optionally start traffic generation with `conda activate carla_app && python -m tools.carla.generate_traffic`

4. Start the producer

5. Start the aggregator

6. Start the visualizer

7. Start one or both detector nodes

## Producer

The producer connects to the active CARLA world, requires manual control to already be running in `--sync` mode, captures front RGB camera and LiDAR snapshots, and writes synchronized frame payloads into shared memory for the downstream nodes.

#### Flags

* `--every-nth` — publish every N-th CARLA frame (default: `1`)
* `--verbose` — print per-frame logs

#### Examples

```bash
conda activate carla_app

# Publish every synchronized CARLA frame.
python -m tools.streaming.live_producer --every-nth 1

# Publish every 5-th synchronized CARLA frame and print logs.
python -m tools.streaming.live_producer \
  --every-nth 5 \
  --verbose
```

## Aggregator

The aggregator reads the shared frame stream, merges object and lane predictions when those branches are active, and publishes one scene stream over ZMQ for the live viewer.

#### Flags

* `--verbose` — print per-frame logs

#### Examples

```bash
conda activate carla_app

# Merge the currently active streaming branches into one scene stream and print logs.
python -m tools.streaming.live_aggregator --verbose
```

## Visualizer

The visualizer subscribes to the aggregated scene stream, renders the current state in Rerun, and computes display-side latency from the embedded publication timestamp.

#### Flags

* `--show-grid` — show the 3D ground grid
* `--verbose` — print per-frame logs

#### Examples

```bash
conda activate carla_app

# Launch the live Rerun viewer and print logs
python -m tools.streaming.live_visualizer --verbose
```

## OpenPCDet Inference Node

This node reads shared LiDAR frames from the producer, runs OpenPCDet inference on each new frame, and writes 3D object predictions back into shared memory for the aggregator.

#### Flags

* `--cfg-file` — OpenPCDet config file used for live inference
* `--ckpt` — checkpoint file used for live inference
* `--score-thresh` — filter low-confidence detections (default: `0.05`)
* `--point-stride` — keep every N-th point before inference (default: `1`)
* `--verbose` — print per-frame logs

#### Examples

```bash
conda activate openpcdet

# Attach OpenPCDet to the live pipeline with a selected config and checkpoint.
python -m tools.streaming.live_openpcdet_inference \
  --cfg-file /path/to/config.yaml \
  --ckpt /path/to/checkpoint.ckpt \
  --verbose
```

## LaneDet Inference Node

This node reads shared front camera frames from the producer, runs LaneDet on each new frame, projects the predicted lanes into 3D, and writes the lane predictions back into shared memory for the aggregator.

#### Flags

* `--config` — LaneDet config file used for live inference
* `--ckpt` — checkpoint file used for live inference
* `--score-thresh` — filter low-confidence lane predictions (default: `0.2`)
* `--verbose` — print per-frame logs

#### Examples

```bash
conda activate lanedet

# Attach LaneDet to the live pipeline with a selected config and checkpoint.
python -m tools.streaming.live_lanedet_inference \
  --config /path/to/config.py \
  --ckpt /path/to/checkpoint.pth \
  --verbose
```
