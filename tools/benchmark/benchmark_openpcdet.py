#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.benchmark.artifacts import build_openpcdet_meta, create_benchmark_output_dir, write_meta_json
from src.benchmark.metrics import summarize_frame_metrics, write_metrics_json
from src.benchmark.predictions import save_objects_prediction
from src.carla.dataset.reader import iter_frame_dirs, load_points_frame
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.openpcdet.model import load_inference_model, run_inference
from src.openpcdet.postprocess import filter_object_predictions


@dataclass(frozen=True)
class BenchmarkArgs:
    run_dir: Path
    cfg_file: Path
    ckpt: Path
    score_thresh: float
    point_stride: int
    warmup: int
    save_pred: bool


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Benchmark OpenPCDet on an offline CARLA raw dataset run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--save-pred", action="store_true", help="Save per-frame predictions")
    parsed = parser.parse_args()
    return BenchmarkArgs(
        run_dir=parsed.run_dir,
        cfg_file=parsed.cfg_file,
        ckpt=parsed.ckpt,
        score_thresh=float(parsed.score_thresh),
        point_stride=int(parsed.point_stride),
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
    if args.point_stride < 1:
        raise ValueError("--point-stride must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    frame_dirs = iter_frame_dirs(args.run_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.run_dir}")

    output_dir = create_benchmark_output_dir(run_dir=args.run_dir, model_name="openpcdet")
    predictions_dir = output_dir / "predictions"

    dataset, model, cfg, logger = load_inference_model(args.cfg_file, args.ckpt)
    logger.info("=== OpenPCDet offline benchmark ===")
    logger.info("run_dir: %s", args.run_dir)
    logger.info("frames: %d", len(frame_dirs))
    logger.info("warmup: %d", args.warmup)
    logger.info("point_stride: %d", args.point_stride)
    logger.info("score_thresh: %.2f", args.score_thresh)
    logger.info("output_dir: %s", output_dir)

    metrics: List[Dict[str, float]] = []
    first_frame_meta = None
    for index, frame_dir in enumerate(tqdm(frame_dirs, desc="Benchmarking OpenPCDet", unit="frame")):
        frame = load_points_frame(frame_dir)

        if first_frame_meta is None:
            first_frame_meta = frame.meta

        t0 = now_synchronized()

        points = np.asarray(frame.points, dtype=np.float32)
        if args.point_stride > 1:
            points = points[:: args.point_stride]

        frame_id = int(frame.meta.get("frame_index", index))

        raw_objects_3d, model_forward_s = run_inference(dataset, model, points, frame_id, return_forward_time=True)

        objects_3d = filter_object_predictions(
            objects_3d=raw_objects_3d,
            class_names=cfg.CLASS_NAMES,
            allowed_classes=NUSCENES_LIKE_CLASSES,
            score_thresh=args.score_thresh,
        )

        t1 = now_synchronized()

        if args.save_pred:
            save_objects_prediction(
                predictions_dir / f"{frame_dir.name}.npz",
                objects_3d=objects_3d,
            )

        item: Dict[str, float] = {
            "frame": float(frame_id),
            "warmup": float(index < args.warmup),
            "model_forward_ms": 1000.0 * model_forward_s,
            "runtime_ms": 1000.0 * (t1 - t0),
            "num_points": float(points.shape[0]),
            "num_predictions": float(len(objects_3d)),
        }
        metrics.append(item)

    summary = summarize_frame_metrics(metrics, model="openpcdet", warmup=args.warmup)
    write_metrics_json(output_dir / "metrics.json", summary=summary)
    write_meta_json(
        output_dir / "meta.json",
        payload=build_openpcdet_meta(
            run_dir=args.run_dir,
            cfg_file=args.cfg_file,
            ckpt=args.ckpt,
            frames_total=len(frame_dirs),
            warmup=args.warmup,
            score_thresh=args.score_thresh,
            point_stride=args.point_stride,
            save_predictions=args.save_pred,
            model_class_names=cfg.CLASS_NAMES,
            dataset_meta=first_frame_meta or {},
            created_at=output_dir.name,
        ),
    )
    logger.info("model FPS: %.2f", summary.get("model_fps", 0.0))
    logger.info("runtime FPS: %.2f", summary.get("runtime_fps", 0.0))
    logger.info("results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
