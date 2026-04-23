import numpy as np

import carla
from src.common.config import CameraConfig


def configure_front_camera_blueprint(
    world: carla.World,
    config: CameraConfig,
    fixed_delta_seconds: float,
) -> carla.ActorBlueprint:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(config.width))
    blueprint.set_attribute("image_size_y", str(config.height))
    blueprint.set_attribute("fov", str(config.fov))
    blueprint.set_attribute("sensor_tick", str(fixed_delta_seconds))
    return blueprint


def front_camera_transform(config: CameraConfig) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=config.x, y=config.y, z=config.z),
        carla.Rotation(pitch=config.pitch, yaw=config.yaw, roll=config.roll),
    )


def camera_image_to_bgr(image: carla.Image) -> np.ndarray:
    data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return data[:, :, :3].copy()
