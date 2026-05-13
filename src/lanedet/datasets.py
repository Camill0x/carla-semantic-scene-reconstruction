import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.common.dataset import DatasetSplits
from src.common.typing_aliases import JsonDict

TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))
TRAIN_FILES = ("label_data_0313.json", "label_data_0601.json")
VAL_FILE = "label_data_0531.json"
TEST_FILE = "test_label.json"

LanePoint = Tuple[float, float]


def load_lane_points(lanes_path: Path) -> List[List[LanePoint]]:
    """Load lane point polylines from a saved lane-annotation file."""
    with lanes_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return []

    lanes: List[List[LanePoint]] = []
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        points = lane.get("points", [])
        clean_points = [(float(point[0]), float(point[1])) for point in points if len(point) >= 2]
        if len(clean_points) >= 2:
            lanes.append(clean_points)
    return lanes


def interpolate_lane_xs(points: Sequence[Tuple[float, float]], h_samples: Sequence[int]) -> List[int]:
    """Interpolate lane x coordinates for the requested TuSimple y samples."""
    ordered = sorted(points, key=lambda point: point[1])
    deduped = []
    used_y = set()
    for x, y in ordered:
        y_key = round(y, 3)
        if y_key in used_y:
            continue
        used_y.add(y_key)
        deduped.append((x, y))

    if len(deduped) < 2:
        return [-2 for _ in h_samples]

    xs = np.asarray([point[0] for point in deduped], dtype=np.float64)
    ys = np.asarray([point[1] for point in deduped], dtype=np.float64)
    sampled = []
    for y in h_samples:
        if y < ys[0] or y > ys[-1]:
            sampled.append(-2)
            continue
        x = float(np.interp(float(y), ys, xs))
        sampled.append(int(round(x)) if x >= 0 else -2)
    return sampled


def lane_bottom_x(lane: Sequence[int], h_samples: Sequence[int]) -> float:
    """Return the x position of the lowest valid sample in a lane."""
    for x, _ in sorted(zip(lane, h_samples), key=lambda item: item[1], reverse=True):
        if x >= 0:
            return float(x)
    return float("inf")


def sample_lanes(
    lane_points: Sequence[Sequence[Tuple[float, float]]],
    h_samples: Sequence[int],
    max_lanes: int,
) -> List[List[int]]:
    """Convert lane point polylines into TuSimple-sampled lane arrays."""
    sampled = [interpolate_lane_xs(points, h_samples) for points in lane_points]
    sampled = [lane for lane in sampled if sum(1 for x in lane if x >= 0) >= 2]
    sampled.sort(key=lambda lane: lane_bottom_x(lane, h_samples))
    return sampled[:max_lanes]


def frame_meta_num_lanes(frame_dir: Path) -> Optional[int]:
    """Read the number of collected lanes from frame metadata when available."""
    meta_path = frame_dir / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if "num_lanes" not in meta:
        return None
    return int(meta["num_lanes"])


def frame_to_sample(frame_dir: Path, max_lanes: int) -> Optional[JsonDict]:
    """Convert one recorded CARLA frame into a LaneDet training sample."""
    image_path = frame_dir / "front_rgb.png"
    lanes_path = frame_dir / "lanes.json"
    if not image_path.exists() or not lanes_path.exists():
        return None

    lane_points = load_lane_points(lanes_path)
    lanes = sample_lanes(lane_points, TUSIMPLE_H_SAMPLES, max_lanes=max_lanes)
    if not lanes:
        return None

    sample_id = f"{frame_dir.parent.name}__{frame_dir.name}"
    raw_file = f"clips/{frame_dir.parent.name}/{frame_dir.name}.png"
    return {
        "sample_id": sample_id,
        "source_image_path": image_path,
        "raw_file": raw_file,
        "lanes": lanes,
        "h_samples": TUSIMPLE_H_SAMPLES,
    }


def load_samples(
    frame_dirs: Sequence[Path],
    max_lanes: int,
    show_progress: bool,
) -> Tuple[List[JsonDict], Dict[str, int]]:
    """Load LaneDet training samples from the selected frame directories."""
    samples: List[JsonDict] = []
    skipped_missing_files = 0
    skipped_no_lanes_meta = 0
    skipped_no_usable_lanes = 0

    iterator = tqdm(frame_dirs, desc="Processing frames", unit="frame") if show_progress else frame_dirs
    for frame_dir in iterator:
        image_path = frame_dir / "front_rgb.png"
        lanes_path = frame_dir / "lanes.json"
        if not image_path.exists() or not lanes_path.exists():
            skipped_missing_files += 1
            continue

        if frame_meta_num_lanes(frame_dir) == 0:
            skipped_no_lanes_meta += 1
            continue

        sample = frame_to_sample(frame_dir, max_lanes=max_lanes)
        if sample is not None:
            samples.append(sample)
        else:
            skipped_no_usable_lanes += 1

    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate LaneDet sample ids detected after dataset preparation")

    stats = {
        "total_frames": len(frame_dirs),
        "usable_samples": len(samples),
        "skipped_missing_files": skipped_missing_files,
        "skipped_no_lanes_meta": skipped_no_lanes_meta,
        "skipped_no_usable_lanes": skipped_no_usable_lanes,
    }
    return samples, stats


def tusimple_payload(sample: JsonDict) -> JsonDict:
    """Convert an internal lane sample into the TuSimple JSON-lines format."""
    return {
        "lanes": sample["lanes"],
        "h_samples": sample["h_samples"],
        "raw_file": sample["raw_file"],
    }


def write_json_lines(path: Path, samples: Iterable[JsonDict]) -> None:
    """Write a sequence of JSON objects as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(tusimple_payload(sample), separators=(",", ":")))
            handle.write("\n")


def write_train_files(output_root: Path, train_samples: Sequence[JsonDict]) -> None:
    """Write the LaneDet training label files for the prepared dataset."""
    chunks: List[List[JsonDict]] = [[], []]
    for index, sample in enumerate(train_samples):
        chunks[index % len(chunks)].append(sample)

    for filename, samples in zip(TRAIN_FILES, chunks):
        write_json_lines(output_root / filename, samples)


def draw_segmentation_mask(sample: JsonDict, output_root: Path, line_width: int) -> None:
    """Render a lane sample into its segmentation mask image."""
    image = cv2.imread(str(sample["source_image_path"]))
    if image is None:
        raise FileNotFoundError(sample["source_image_path"])

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lane_index, lane in enumerate(sample["lanes"]):
        points = [
            (int(x), int(y))
            for x, y in zip(lane, sample["h_samples"])
            if x >= 0 and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]
        ]
        if len(points) < 2:
            continue
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(mask, start, end, color=lane_index + 1, thickness=line_width)

    mask_path = output_root / sample["raw_file"].replace("clips", "seg_label")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)


def materialize_samples(
    output_root: Path,
    samples: Sequence[JsonDict],
    line_width: int,
    show_progress: bool,
) -> None:
    """Write image and mask assets for a collection of LaneDet samples."""
    iterator = tqdm(samples, desc="Writing samples", unit="sample") if show_progress else samples
    for sample in iterator:
        image_target = output_root / sample["raw_file"]
        image_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["source_image_path"], image_target)
        draw_segmentation_mask(sample, output_root, line_width=line_width)


def write_tusimple_dataset(
    output_root: Path,
    splits: DatasetSplits[JsonDict],
    line_width: int,
    show_progress: bool = True,
) -> None:
    """Write the prepared LaneDet dataset in TuSimple-compatible format."""
    all_samples = [*splits.train, *splits.val, *splits.test]

    materialize_samples(output_root, all_samples, line_width=line_width, show_progress=show_progress)
    write_train_files(output_root, splits.train)
    write_json_lines(output_root / VAL_FILE, splits.val)
    write_json_lines(output_root / TEST_FILE, splits.test)
