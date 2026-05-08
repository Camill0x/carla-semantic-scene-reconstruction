#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.benchmark.artifacts import build_openpcdet_meta, create_benchmark_output_dir, write_meta_json
from src.benchmark.metrics import summarize_frame_metrics, write_metrics_json
from src.benchmark.predictions import save_objects_prediction
from src.carla.dataset.reader import iter_frame_dirs, load_points_frame
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.openpcdet.model import load_inference_model
from src.openpcdet.predict import filter_object_predictions, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OpenPCDet on an offline CARLA raw dataset run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-pred", action="store_true", help="Save per-frame predictions")
    return parser.parse_args()


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
    if args.limit is not None:
        frame_dirs = frame_dirs[: args.limit]
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
    logger.info("score_thresh: %.3f", args.score_thresh)
    logger.info("output_dir: %s", output_dir)

    metrics = []
    first_frame_meta = None
    for index, frame_dir in enumerate(tqdm(frame_dirs, desc="Benchmarking OpenPCDet", unit="frame")):
        t0 = time.perf_counter()

        frame = load_points_frame(frame_dir)

        t1 = time.perf_counter()

        if first_frame_meta is None:
            first_frame_meta = frame.meta

        points = np.asarray(frame.points, dtype=np.float32)
        if args.point_stride > 1:
            points = points[:: args.point_stride]

        frame_id = int(frame.meta.get("frame_index", index))

        t2 = now_synchronized()

        pred_dict = run_inference(dataset, model, points, frame_id)

        t3 = now_synchronized()

        prediction = filter_object_predictions(
            pred_dict=pred_dict,
            class_names=cfg.CLASS_NAMES,
            allowed_classes=NUSCENES_LIKE_CLASSES,
            score_thresh=args.score_thresh,
        )

        t4 = time.perf_counter()

        if args.save_pred:
            save_objects_prediction(
                predictions_dir / f"{frame_dir.name}.npz",
                boxes=prediction.boxes,
                scores=prediction.scores,
                names=prediction.names,
            )

        item = {
            "frame": frame_id,
            "warmup": bool(index < args.warmup),
            "io_ms": 1000.0 * (t1 - t0),
            "preprocess_ms": 0.0,
            "infer_ms": 1000.0 * (t3 - t2),
            "postprocess_ms": 1000.0 * (t4 - t3),
            "total_ms": 1000.0 * (t4 - t2),
            "num_points": int(points.shape[0]),
            "num_predictions": len(prediction.names),
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
            limit=args.limit,
            score_thresh=args.score_thresh,
            point_stride=args.point_stride,
            save_predictions=args.save_pred,
            model_class_names=cfg.CLASS_NAMES,
            dataset_meta=first_frame_meta or {},
            created_at=output_dir.name,
        ),
    )
    logger.info("inference FPS: %.2f", summary.get("inference_fps", 0.0))
    logger.info("results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
