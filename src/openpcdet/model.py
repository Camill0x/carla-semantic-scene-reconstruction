import os
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils

from src.openpcdet.runtime_dataset import RealtimeDataset


@contextmanager
def working_directory(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def get_openpcdet_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "OpenPCDet"


def load_inference_model(cfg_file: Path, ckpt: Path) -> Tuple[RealtimeDataset, torch.nn.Module, object, object]:
    cfg_file_path = cfg_file.expanduser().resolve()
    ckpt_path = ckpt.expanduser().resolve()
    openpcdet_root = get_openpcdet_root()

    with working_directory(openpcdet_root):
        cfg_from_yaml_file(str(cfg_file_path), cfg)
    logger = common_utils.create_logger()

    dataset = RealtimeDataset(
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
