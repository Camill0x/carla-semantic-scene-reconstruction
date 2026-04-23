from typing import Dict, Sequence

import numpy as np
import open3d as o3d

from src.carla.geometry.transforms import world_to_sensor_dict
from src.carla.vis.boxes import bbox_to_world_corners, gt_box7_to_corners_sensor
from src.carla.vis.colors import EGO_COLOR, get_object_color


def make_bbox_lineset(corners: np.ndarray, color: Sequence[float]) -> o3d.geometry.LineSet:
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set


def make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    xyz = points[:, :3]
    intensity = points[:, 3].astype(np.float32)

    if intensity.size > 0 and intensity.max() > intensity.min():
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    else:
        intensity_norm = np.zeros_like(intensity)

    colors = np.stack(
        [
            0.10 + 0.70 * intensity_norm,
            0.15 + 0.75 * intensity_norm,
            0.20 + 0.80 * intensity_norm,
        ],
        axis=1,
    )
    colors = np.clip(colors, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_geometries(
    lidar_transform: Dict,
    points: np.ndarray,
    ego_box: np.ndarray,
    objects: Sequence[Dict],
):
    geometries = [make_point_cloud(points)]

    sensor_location = lidar_transform["location"]
    sensor_rotation = lidar_transform["rotation"]

    for obj in objects:
        world_corners = bbox_to_world_corners(obj)
        sensor_corners = world_to_sensor_dict(world_corners, sensor_location, sensor_rotation)
        sensor_corners[:, 1] *= -1.0

        color = get_object_color(obj)
        geometries.append(make_bbox_lineset(sensor_corners, color))

    if ego_box is not None and ego_box.shape[0] == 7:
        ego_corners = gt_box7_to_corners_sensor(ego_box)
        geometries.append(make_bbox_lineset(ego_corners, EGO_COLOR))

    return geometries


def show_frame(
        lidar_transform: Dict,
        points: np.ndarray,
        ego_box: np.ndarray,
        objects: Sequence[Dict],
) -> None:
    geometries = build_geometries(lidar_transform, points, ego_box, objects)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Detector frame viewer",
        width=1280,
        height=720,
        visible=True,
    )

    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    render_option.point_size = 1.5
    render_option.line_width = 2.0

    view_ctl = vis.get_view_control()
    view_ctl.set_front(np.array([-1.0, 0.0, 0.3]))
    view_ctl.set_up(np.array([0.0, 0.0, 1.0]))
    view_ctl.set_lookat(np.array([0.0, 0.0, 0.0]))
    view_ctl.set_zoom(0.1)

    vis.run()
    vis.destroy_window()
