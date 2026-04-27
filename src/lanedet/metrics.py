import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TUSIMPLE_PIXEL_THRESH = 20.0
TUSIMPLE_POINT_THRESH = 0.85


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def build_gt_map(gt_path: Path) -> Dict[str, dict]:
    return {record["raw_file"]: record for record in load_jsonl(gt_path) if record.get("raw_file")}


def valid_pairs(gt_lane: List[int], pred_lane: List[int], h_samples: List[int]) -> List[Tuple[int, int, int]]:
    pairs = []
    for x_gt, x_pred, y in zip(gt_lane, pred_lane, h_samples):
        if x_gt is None or x_pred is None:
            continue
        if x_gt < 0 or x_pred < 0:
            continue
        pairs.append((int(x_gt), int(x_pred), int(y)))
    return pairs


def lane_mae(gt_lane: List[int], pred_lane: List[int], h_samples: List[int]) -> Optional[float]:
    pairs = valid_pairs(gt_lane, pred_lane, h_samples)
    if not pairs:
        return None
    return sum(abs(x_gt - x_pred) for x_gt, x_pred, _ in pairs) / len(pairs)


def lane_rmse(gt_lane: List[int], pred_lane: List[int], h_samples: List[int]) -> Optional[float]:
    pairs = valid_pairs(gt_lane, pred_lane, h_samples)
    if not pairs:
        return None
    return math.sqrt(sum((x_gt - x_pred) ** 2 for x_gt, x_pred, _ in pairs) / len(pairs))


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def tusimple_angle(xs: np.ndarray, y_samples: np.ndarray) -> float:
    xs = np.asarray(xs)
    y_samples = np.asarray(y_samples)
    valid = xs >= 0
    xs = xs[valid]
    ys = y_samples[valid]
    if len(xs) <= 1:
        return 0.0
    y_mean = float(np.mean(ys))
    x_mean = float(np.mean(xs))
    denom = float(np.sum((ys - y_mean) ** 2))
    slope = 0.0 if denom == 0.0 else float(np.sum((ys - y_mean) * (xs - x_mean)) / denom)
    return float(np.arctan(slope))


def tusimple_line_accuracy(pred_lane: List[int], gt_lane: List[int], threshold: float) -> float:
    pred = np.array([x if x >= 0 else -100 for x in pred_lane])
    gt = np.array([x if x >= 0 else -100 for x in gt_lane])
    return float(np.sum(np.where(np.abs(pred - gt) < threshold, 1.0, 0.0)) / len(gt))


def analyze_predictions(
    gt_map: Dict[str, dict],
    pred_records: List[dict],
) -> dict:
    n_images = 0
    total_gt_lanes = 0
    total_pred_lanes = 0
    accuracy_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    point_errors = []
    matched_lane_maes = []
    matched_lane_rmses = []

    for pred in pred_records:
        if "raw_file" not in pred or "lanes" not in pred or "run_time" not in pred:
            raise ValueError("raw_file or lanes or run_time not in some predictions.")

        raw_file = pred.get("raw_file")
        gt = gt_map.get(raw_file)
        if gt is None:
            raise ValueError("Some raw_file from your predictions do not exist in the test tasks.")

        n_images += 1
        gt_lanes = gt.get("lanes", [])
        pred_lanes = pred.get("lanes", [])
        h_samples = gt.get("h_samples", [])
        total_gt_lanes += len(gt_lanes)
        total_pred_lanes += len(pred_lanes)
        run_time = float(pred.get("run_time", 1.0))

        if run_time > 200 or len(gt_lanes) + 2 < len(pred_lanes):
            fp_sum += 0.0
            fn_sum += 1.0
            continue

        angles = [tusimple_angle(np.array(gt_lane), np.array(h_samples)) for gt_lane in gt_lanes]
        thresholds = [TUSIMPLE_PIXEL_THRESH / np.cos(angle) for angle in angles]
        line_accuracies = []
        line_matches = []
        image_fn = 0.0
        image_matched = 0.0

        for gt_idx, (gt_lane, threshold) in enumerate(zip(gt_lanes, thresholds)):
            accuracies = [
                tusimple_line_accuracy(pred_lane, gt_lane, threshold)
                for pred_lane in pred_lanes
            ]
            best_accuracy = max(accuracies) if accuracies else 0.0
            best_pred_idx = accuracies.index(best_accuracy) if accuracies else None
            line_accuracies.append(best_accuracy)

            if best_accuracy < TUSIMPLE_POINT_THRESH or best_pred_idx is None:
                image_fn += 1.0
                continue

            image_matched += 1.0
            line_matches.append((best_accuracy, gt_idx, best_pred_idx))

        image_fp = len(pred_lanes) - image_matched
        if len(gt_lanes) > 4 and image_fn > 0:
            image_fn -= 1.0

        accuracy_total = sum(line_accuracies)
        ignored_accuracy = min(line_accuracies) if len(gt_lanes) > 4 and line_accuracies else None
        if ignored_accuracy is not None:
            accuracy_total -= ignored_accuracy

        accuracy_sum += accuracy_total / max(min(4.0, len(gt_lanes)), 1.0)
        fp_sum += image_fp / len(pred_lanes) if pred_lanes else 0.0
        fn_sum += image_fn / max(min(len(gt_lanes), 4.0), 1.0)

        for best_accuracy, gt_idx, pred_idx in line_matches:
            if ignored_accuracy is not None and best_accuracy == ignored_accuracy:
                ignored_accuracy = None
                continue

            mae = lane_mae(gt_lanes[gt_idx], pred_lanes[pred_idx], h_samples)
            if mae is not None:
                matched_lane_maes.append(mae)

            rmse = lane_rmse(gt_lanes[gt_idx], pred_lanes[pred_idx], h_samples)
            if rmse is not None:
                matched_lane_rmses.append(rmse)

            for x_gt, x_pred, _ in valid_pairs(gt_lanes[gt_idx], pred_lanes[pred_idx], h_samples):
                error = abs(x_gt - x_pred)
                point_errors.append(error)

    accuracy = accuracy_sum / n_images if n_images else 0.0
    fp = fp_sum / n_images if n_images else 0.0
    fn = fn_sum / n_images if n_images else 0.0

    return {
        "images_evaluated": n_images,
        "total_gt_lanes": total_gt_lanes,
        "total_pred_lanes": total_pred_lanes,
        "accuracy": rounded(accuracy),
        "fp": rounded(fp),
        "fn": rounded(fn),
        "matched_lane_mae_px": rounded(
            sum(matched_lane_maes) / len(matched_lane_maes) if matched_lane_maes else None,
            digits=2,
        ),
        "matched_lane_rmse_px": rounded(
            sum(matched_lane_rmses) / len(matched_lane_rmses) if matched_lane_rmses else None,
            digits=2,
        ),
        "point_mae_px": rounded(
            sum(point_errors) / len(point_errors) if point_errors else None,
            digits=2,
        ),
        "point_rmse_px": rounded(
            math.sqrt(sum(error * error for error in point_errors) / len(point_errors)) if point_errors else None,
            digits=2,
        ),
    }


def build_tusimple_metrics(pred_json: Path, gt_json: Path) -> dict:
    pred_records = load_jsonl(pred_json)
    gt_records = load_jsonl(gt_json)
    if len(gt_records) != len(pred_records):
        raise ValueError("We do not get the predictions of all the test tasks")

    gt_map = build_gt_map(gt_json)
    return analyze_predictions(gt_map=gt_map, pred_records=pred_records)
