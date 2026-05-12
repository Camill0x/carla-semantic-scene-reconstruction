#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from src.common.cli_logging import print_verbose
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.common.runtime_config import build_streaming_openpcdet_inference_config
from src.openpcdet.model import load_inference_model, run_inference
from src.openpcdet.postprocess import filter_object_predictions
from src.shared_memory.buffers import SharedArrayReader, SharedMessageBuffer
from src.shared_memory.names import SharedMemoryNames
from src.streaming.messages import build_objects_3d_frame_message, parse_frame_snapshot_message


def parse_args():
    parser = argparse.ArgumentParser(description="OpenPCDet live inference node")
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Score threshold for predictions")
    parser.add_argument("--point-stride", type=int, default=1, help="Take every N-th point before inference")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_streaming_openpcdet_inference_config(
        cfg_file=args.cfg_file,
        ckpt=args.ckpt,
        score_thresh=args.score_thresh,
        point_stride=args.point_stride,
    )
    names = SharedMemoryNames(prefix=config.common.prefix)

    dataset, model, cfg, logger = load_inference_model(config.cfg_file, config.ckpt)
    frame_buffer = SharedMessageBuffer(
        name=names.frame_buffer,
        size_bytes=config.common.frame_buffer_size_bytes,
        create=False,
    )
    objects_buffer = SharedMessageBuffer(
        name=names.objects_buffer,
        size_bytes=config.common.objects_buffer_size_bytes,
        create=False,
    )
    reader = SharedArrayReader()
    last_version = None
    last_frame = None
    sleep_s = max(0.001, config.common.poll_interval_ms / 1000.0)

    logger.info("=== OpenPCDet streaming inference ===")
    logger.info("frame buffer: %s", names.frame_buffer)
    logger.info("objects buffer: %s", names.objects_buffer)

    try:
        while True:
            version, payload = frame_buffer.read(last_version=last_version)
            if payload is None:
                time.sleep(sleep_s)
                continue

            last_version = version
            try:
                frame_message = parse_frame_snapshot_message(payload)
            except Exception as exc:
                logger.warning("Skipping invalid frame snapshot: %s", exc)
                continue

            frame_id = frame_message["frame"]
            if frame_id == last_frame:
                continue

            try:
                points4 = reader.read(frame_message["lidar"]["shared_array"])
                if config.point_stride > 1:
                    points4 = points4[:: config.point_stride]

                raw_objects_3d = run_inference(dataset, model, points4, frame_id)
                objects_3d = filter_object_predictions(
                    objects_3d=raw_objects_3d,
                    class_names=cfg.CLASS_NAMES,
                    allowed_classes=NUSCENES_LIKE_CLASSES,
                    score_thresh=config.score_thresh,
                )
            except Exception as exc:
                logger.exception("Inference failed frame=%s: %s", frame_id, exc)
                continue

            objects_buffer.write(
                build_objects_3d_frame_message(
                    lidar_message=frame_message,
                    objects_3d=objects_3d,
                )
            )
            print_verbose(args.verbose, "OpenPCDet", f"Detected {len(objects_3d)} objects for frame {frame_id}")
            last_frame = frame_id
    finally:
        reader.close()
        frame_buffer.close()
        objects_buffer.close()


if __name__ == "__main__":
    main()
