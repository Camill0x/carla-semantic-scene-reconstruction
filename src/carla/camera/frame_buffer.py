import queue
from typing import Optional, Tuple

import numpy as np

import carla
from src.carla.camera.sensor import camera_image_to_bgr


class CameraFrameBuffer:
    def __init__(self, camera: carla.Sensor):
        self.camera = camera
        self.queue: "queue.Queue[carla.Image]" = queue.Queue()
        self.pending_image: Optional[carla.Image] = None

    def callback(self, image: carla.Image) -> None:
        self.queue.put(image)

    def get_frame(
        self,
        expected_frame: Optional[int] = None,
        timeout: float = 2.0,
        allow_future: bool = False,
    ) -> Tuple[np.ndarray, int, float, carla.Transform]:
        while True:
            if self.pending_image is not None:
                image = self.pending_image
                self.pending_image = None
            else:
                image = self.queue.get(timeout=timeout)
            frame = int(image.frame)

            if expected_frame is None:
                return camera_image_to_bgr(image), frame, float(image.timestamp), image.transform

            if frame < expected_frame:
                continue

            if frame > expected_frame:
                if not allow_future:
                    self.pending_image = image
                    raise RuntimeError(f"Camera frame mismatch: expected {expected_frame}, got {frame}")
                self.pending_image = None
                return camera_image_to_bgr(image), frame, float(image.timestamp), image.transform

            return camera_image_to_bgr(image), frame, float(image.timestamp), image.transform
