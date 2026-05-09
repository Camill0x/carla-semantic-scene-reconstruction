#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.benchmark.artifacts import build_lanedet_meta, create_benchmark_output_dir, write_meta_json
from src.benchmark.metrics import summarize_frame_metrics, write_metrics_json
from src.benchmark.offline_frames import camera_frame_shape, state_frame_from_meta
from src.benchmark.predictions import save_lanes_prediction
from src.carla.dataset.reader import iter_frame_dirs, load_camera_frame
from src.lanedet.detector import LaneDetector
from src.lanedet.projection import lanes_2d_to_lanes_3d_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LaneDet on an offline CARLA raw dataset run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--config", type=Path, required=True, help="LaneDet config file")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.2)
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
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    frame_dirs = iter_frame_dirs(args.run_dir)
    if args.limit is not None:
        frame_dirs = frame_dirs[: args.limit]
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.run_dir}")

    output_dir = create_benchmark_output_dir(run_dir=args.run_dir, model_name="lanedet")
    predictions_dir = output_dir / "predictions"
    detector = LaneDetector(args.config, args.ckpt, score_thresh=args.score_thresh)

    print("=== LaneDet offline benchmark ===")
    print(f"[info] run_dir: {args.run_dir}")
    print(f"[info] frames: {len(frame_dirs)}")
    print(f"[info] warmup: {args.warmup}")
    print(f"[info] score_thresh: {args.score_thresh:.2f}")
    print(f"[info] output_dir: {output_dir}")

    metrics = []
    first_frame_meta = None
    for index, frame_dir in enumerate(tqdm(frame_dirs, desc="Benchmarking LaneDet", unit="frame")):
        frame = load_camera_frame(frame_dir)

        if first_frame_meta is None:
            first_frame_meta = frame.meta

        t0 = now_synchronized()

        image_bgr = np.ascontiguousarray(frame.image_rgb[:, :, ::-1])

        lanes_2d, model_forward_s = detector.infer_lanes_2d(image_bgr, return_forward_time=True)

        lanes_3d = lanes_2d_to_lanes_3d_payload(
            lanes_2d,
            camera_frame=camera_frame_shape(frame.image_rgb),
            state_frame=state_frame_from_meta(frame.meta),
            score_thresh=args.score_thresh,
        )

        t1 = now_synchronized()

        if args.save_pred:
            save_lanes_prediction(predictions_dir / f"{frame_dir.name}.json", lanes_3d)

        frame_id = int(frame.meta.get("frame_index", index))
        num_lanes = len(lanes_3d.get("strips", []))
        item = {
            "frame": frame_id,
            "warmup": bool(index < args.warmup),
            "model_forward_ms": 1000.0 * model_forward_s,
            "runtime_ms": 1000.0 * (t1 - t0),
            "image_height": int(frame.image_rgb.shape[0]),
            "image_width": int(frame.image_rgb.shape[1]),
            "num_predictions": num_lanes,
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
            limit=args.limit,
            score_thresh=args.score_thresh,
            save_predictions=args.save_pred,
            dataset_meta=first_frame_meta or {},
            created_at=output_dir.name,
        ),
    )
    print(f"[info] model FPS: {summary.get('model_fps', 0.0):.2f}")
    print(f"[info] runtime FPS: {summary.get('runtime_fps', 0.0):.2f}")
    print(f"[info] results saved to: {output_dir}")


if __name__ == "__main__":
    main()
