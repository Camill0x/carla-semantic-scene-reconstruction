import carla
from src.common.config import LidarConfig


def configure_lidar_blueprint(
    world: carla.World,
    *,
    config: LidarConfig,
    fixed_delta_seconds: float,
):
    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("upper_fov", str(config.upper_fov))
    lidar_bp.set_attribute("lower_fov", str(config.lower_fov))
    lidar_bp.set_attribute("channels", str(config.channels))
    lidar_bp.set_attribute("range", str(config.max_range))
    lidar_bp.set_attribute("points_per_second", str(config.points_per_second))

    lidar_bp.set_attribute("rotation_frequency", str(1.0 / fixed_delta_seconds))
    lidar_bp.set_attribute("noise_stddev", "0.0")
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
    return lidar_bp
