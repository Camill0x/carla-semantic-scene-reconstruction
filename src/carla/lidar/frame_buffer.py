import queue
from typing import Optional, Tuple

import numpy as np

import carla


class LidarFrameBuffer:
    def __init__(self, hero: carla.Actor, lidar: carla.Sensor):
        self.hero = hero
        self.lidar = lidar
        self.queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()
        self.pending_measurement: Optional[carla.LidarMeasurement] = None

    def callback(self, point_cloud: carla.LidarMeasurement) -> None:
        self.queue.put(point_cloud)

    def discard_before(self, min_frame: int) -> None:
        while True:
            if self.pending_measurement is not None:
                point_cloud = self.pending_measurement
                self.pending_measurement = None
            else:
                try:
                    point_cloud = self.queue.get_nowait()
                except queue.Empty:
                    return

            if int(point_cloud.frame) >= min_frame:
                self.pending_measurement = point_cloud
                return

    def get_frame(
        self,
        expected_frame: Optional[int] = None,
        timeout: float = 2.0,
        allow_future: bool = False,
    ) -> Tuple[np.ndarray, int, float, carla.Transform]:
        while True:
            if self.pending_measurement is not None:
                point_cloud = self.pending_measurement
                self.pending_measurement = None
            else:
                point_cloud = self.queue.get(timeout=timeout)
            frame = int(point_cloud.frame)

            if expected_frame is None:
                points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
                return points, frame, float(point_cloud.timestamp), point_cloud.transform

            if frame < expected_frame:
                continue

            if frame > expected_frame:
                if not allow_future:
                    self.pending_measurement = point_cloud
                    raise RuntimeError(f"Lidar frame mismatch: expected {expected_frame}, got {frame}")
                self.pending_measurement = None
                points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
                return points, frame, float(point_cloud.timestamp), point_cloud.transform

            points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)

            return points, frame, float(point_cloud.timestamp), point_cloud.transform
