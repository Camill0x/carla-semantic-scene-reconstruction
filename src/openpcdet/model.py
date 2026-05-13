import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from src.openpcdet.paths import OPENPCDET_ROOT
from src.openpcdet.prediction import Objects3DPrediction
from src.openpcdet.runner import working_directory


class InferenceDataset(DatasetTemplate):  # type: ignore[misc]
    def __init__(self, dataset_cfg: Any, class_names: Any, logger: Any = None) -> None:
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=None,
            logger=logger,
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("InferenceDataset is only used through prepare_data/collate_batch")


def append_zero_timestamps(points4: Any) -> Any:
    timestamps = np.zeros((points4.shape[0], 1), dtype=np.float32)
    return np.hstack([points4.astype(np.float32), timestamps])


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_inference_model(cfg_file: Path, ckpt: Path) -> Tuple[InferenceDataset, torch.nn.Module, Any, Any]:
    cfg_file_path = cfg_file.expanduser().resolve()
    ckpt_path = ckpt.expanduser().resolve()

    with working_directory(OPENPCDET_ROOT):
        cfg_from_yaml_file(str(cfg_file_path), cfg)
    logger = common_utils.create_logger()

    dataset = InferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        logger=logger,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    return dataset, model, cfg, logger


def run_inference(
    dataset: InferenceDataset,
    model: Any,
    points4: Any,
    frame_id: int,
    *,
    return_forward_time: bool = False,
) -> Any:
    points5 = append_zero_timestamps(points4)
    input_dict = {
        "points": points5,
        "frame_id": frame_id,
    }
    data_dict = dataset.prepare_data(data_dict=input_dict)
    batch_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch_dict)

    if return_forward_time:
        synchronize_cuda()
        t0 = time.perf_counter()

    with torch.no_grad():
        pred_dicts, _ = model.forward(batch_dict)

    if return_forward_time:
        synchronize_cuda()
        forward_time_s = time.perf_counter() - t0

    objects_3d = Objects3DPrediction.from_detector_output(pred_dicts[0])

    if return_forward_time:
        return objects_3d, forward_time_s

    return objects_3d
