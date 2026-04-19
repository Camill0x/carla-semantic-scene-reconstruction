import queue
from typing import Tuple

import numpy as np

import carla


class LidarFrameBuffer:
    def __init__(self, hero: carla.Actor, lidar: carla.Sensor):
        self.hero = hero
        self.lidar = lidar
        self.queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()

    def callback(self, point_cloud: carla.LidarMeasurement) -> None:
        self.queue.put(point_cloud)

    def get_frame(
        self,
        expected_frame: int,
        timeout: float = 2.0,
    ) -> Tuple[np.ndarray, int, float, carla.Transform]:
        while True:
            point_cloud = self.queue.get(timeout=timeout)
            frame = int(point_cloud.frame)

            if frame < expected_frame:
                continue

            if frame > expected_frame:
                raise RuntimeError(f"Lidar frame mismatch: expected {expected_frame}, got {frame}")

            points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)

            return points, frame, float(point_cloud.timestamp), point_cloud.transform
