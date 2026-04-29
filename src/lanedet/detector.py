from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.net_utils import load_network


def _to_device(data: dict, device: torch.device) -> dict:
    out = {}
    for key, value in data.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


class LaneDetector:
    def __init__(self, cfg_file: Path, ckpt: Path, score_thresh: float = None):
        self.cfg = Config.fromfile(str(cfg_file))
        self.cfg.show = False
        self.cfg.savedir = None
        self.cfg.load_from = str(ckpt)
        if score_thresh is not None:
            self.cfg.test_parameters.conf_threshold = float(score_thresh)
        self.processes = Process(self.cfg.val_process, self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = build_net(self.cfg)
        if self.device.type == "cuda":
            net = torch.nn.parallel.DataParallel(net, device_ids=range(1)).cuda()
        else:
            net = net.to(self.device)
        net.eval()
        load_network(net, str(ckpt))
        self.net = net

    def preprocess(self, image_bgr: np.ndarray) -> dict:
        ori_img = np.asarray(image_bgr, dtype=np.uint8)
        img = ori_img[self.cfg.cut_height :, :, :].astype(np.float32)
        data = {"img": img, "lanes": []}
        data = self.processes(data)
        data["img"] = data["img"].unsqueeze(0)
        data.update({"img_path": "<live>", "ori_img": ori_img})
        return _to_device(data, self.device)

    def infer_lanes_2d(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        data = self.preprocess(image_bgr)
        with torch.no_grad():
            output = self.net(data)
            lane_head = self.net.module if hasattr(self.net, "module") else self.net
            lanes = lane_head.get_lanes(output)[0]

        out = []
        for lane in lanes:
            points = np.asarray(lane.to_array(self.cfg), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
                continue
            score = float(lane.metadata.get("conf", 1.0)) if hasattr(lane, "metadata") else 1.0
            out.append((points, score))
        return out
