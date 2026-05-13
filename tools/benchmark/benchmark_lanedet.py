#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.benchmark.artifacts import build_lanedet_meta, create_benchmark_output_dir, write_meta_json
from src.benchmark.frame_payloads import camera_frame_shape, state_frame_from_meta
from src.benchmark.metrics import summarize_frame_metrics, write_metrics_json
from src.benchmark.predictions import save_lanes_prediction
from src.carla.dataset.reader import iter_frame_dirs, load_camera_frame
from src.common.cli_logging import configure_logging
from src.lanedet.model import LaneDetector
from src.lanedet.projection import lanes_2d_to_lanes_3d


@dataclass(frozen=True)
class BenchmarkArgs:
    run_dir: Path
    config: Path
    ckpt: Path
    score_thresh: float
    warmup: int
    save_pred: bool


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Benchmark LaneDet on an offline CARLA raw dataset run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--config", type=Path, required=True, help="LaneDet config file")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--save-pred", action="store_true", help="Save per-frame predictions")
    parsed = parser.parse_args()
    return BenchmarkArgs(
        run_dir=parsed.run_dir,
        config=parsed.config,
        ckpt=parsed.ckpt,
        score_thresh=float(parsed.score_thresh),
        warmup=int(parsed.warmup),
        save_pred=bool(parsed.save_pred),
    )


def cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def now_synchronized() -> float:
    cuda_synchronize()
    return time.perf_counter()


def main() -> None:
    args = parse_args()
    logger = configure_logging("tools.benchmark.lanedet")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    frame_dirs = iter_frame_dirs(args.run_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.run_dir}")

    output_dir = create_benchmark_output_dir(run_dir=args.run_dir, model_name="lanedet")
    predictions_dir = output_dir / "predictions"
    detector = LaneDetector(args.config, args.ckpt, score_thresh=args.score_thresh, logger=logger)

    logger.info("=== LaneDet offline benchmark ===")
    logger.info("run_dir: %s", args.run_dir)
    logger.info("frames: %d", len(frame_dirs))
    logger.info("warmup: %d", args.warmup)
    logger.info("score_thresh: %.2f", args.score_thresh)
    logger.info("output_dir: %s", output_dir)

    metrics: List[Dict[str, float]] = []
    first_frame_meta = None
    for index, frame_dir in enumerate(tqdm(frame_dirs, desc="Benchmarking LaneDet", unit="frame")):
        frame = load_camera_frame(frame_dir)

        if first_frame_meta is None:
            first_frame_meta = frame.meta

        t0 = now_synchronized()

        image_bgr = np.ascontiguousarray(frame.image_rgb[:, :, ::-1])

        lanes_2d, model_forward_s = detector.infer_lanes_2d(image_bgr, return_forward_time=True)

        lanes_3d = lanes_2d_to_lanes_3d(
            lanes_2d,
            camera_frame=camera_frame_shape(frame.image_rgb),
            state_frame=state_frame_from_meta(frame.meta),
            score_thresh=args.score_thresh,
        )

        t1 = now_synchronized()

        if args.save_pred:
            save_lanes_prediction(
                predictions_dir / f"{frame_dir.name}.json",
                lanes_2d=lanes_2d,
                lanes_3d=lanes_3d,
            )

        frame_id = int(frame.meta.get("frame_index", index))
        num_lanes = len(lanes_3d)
        item: Dict[str, float] = {
            "frame": float(frame_id),
            "warmup": float(index < args.warmup),
            "model_forward_ms": 1000.0 * model_forward_s,
            "runtime_ms": 1000.0 * (t1 - t0),
            "image_height": float(frame.image_rgb.shape[0]),
            "image_width": float(frame.image_rgb.shape[1]),
            "num_predictions": float(num_lanes),
        }
        metrics.append(item)

    summary = summarize_frame_metrics(metrics, model="lanedet", warmup=args.warmup)
    write_metrics_json(output_dir / "metrics.json", summary=summary)
    write_meta_json(
        output_dir / "meta.json",
        payload=build_lanedet_meta(
            run_dir=args.run_dir,
            config=args.config,
            ckpt=args.ckpt,
            frames_total=len(frame_dirs),
            warmup=args.warmup,
            score_thresh=args.score_thresh,
            save_predictions=args.save_pred,
            dataset_meta=first_frame_meta or {},
            created_at=output_dir.name,
        ),
    )
    logger.info("model FPS: %.2f", summary.get("model_fps", 0.0))
    logger.info("runtime FPS: %.2f", summary.get("runtime_fps", 0.0))
    logger.info("results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
