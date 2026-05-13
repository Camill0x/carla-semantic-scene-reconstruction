import time
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.net_utils import load_network
from src.common.typing_aliases import ImageArray, JsonDict
from src.lanedet.prediction import Lanes2DPrediction


def _to_device(data: JsonDict, device: torch.device) -> JsonDict:
    out: JsonDict = {}
    for key, value in data.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class LaneDetector:
    def __init__(self, cfg_file: Path, ckpt: Path, score_thresh: float, logger: Logger) -> None:
        self.cfg = Config.fromfile(str(cfg_file))
        self.cfg.show = False
        self.cfg.savedir = None
        self.cfg.load_from = str(ckpt)
        self.cfg.test_parameters.conf_threshold = float(score_thresh)
        self.processes = Process(self.cfg.val_process, self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = build_net(self.cfg)
        if self.device.type == "cuda":
            net = torch.nn.parallel.DataParallel(net, device_ids=range(1)).cuda()
        else:
            net = net.to(self.device)
        net.eval()
        load_network(net, str(ckpt), logger=logger)
        self.net = net

    def preprocess(self, image_bgr: ImageArray) -> JsonDict:
        ori_img = np.asarray(image_bgr, dtype=np.uint8)
        img = ori_img[self.cfg.cut_height :, :, :].astype(np.float32)
        data: JsonDict = {"img": img, "lanes": []}
        data = self.processes(data)
        data["img"] = data["img"].unsqueeze(0)
        data.update({"img_path": "<live>", "ori_img": ori_img})
        return _to_device(data, self.device)

    def infer_lanes_2d(self, image_bgr: ImageArray, *, return_forward_time: bool = False) -> Any:
        data = self.preprocess(image_bgr)

        if return_forward_time:
            _synchronize_cuda()
            t0 = time.perf_counter()

        with torch.no_grad():
            output = self.net(data)

        if return_forward_time:
            _synchronize_cuda()
            forward_time_s = time.perf_counter() - t0

        with torch.no_grad():
            lane_head = self.net.module if hasattr(self.net, "module") else self.net
            lanes = lane_head.get_lanes(output)[0]

        out = []
        for lane in lanes:
            points = np.asarray(lane.to_array(self.cfg), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
                continue
            score = float(lane.metadata.get("conf", 1.0)) if hasattr(lane, "metadata") else 1.0
            out.append((points, score))

        lanes_2d = Lanes2DPrediction.from_detector_output(out)

        if return_forward_time:
            return lanes_2d, forward_time_s

        return lanes_2d
