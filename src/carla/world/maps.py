from typing import List, Tuple

import carla


def _map_basename(map_path: str) -> str:
    return map_path.rsplit("/", 1)[-1]


def _available_map_names(client: carla.Client) -> List[str]:
    available_maps = client.get_available_maps()
    return sorted({_map_basename(map_path) for map_path in available_maps})


def ensure_map_name_exists(client: carla.Client, requested_map: str) -> None:
    available_map_names = _available_map_names(client)
    if requested_map in available_map_names:
        return

    raise ValueError("Map %r does not exist. Choose one of: %s" % (requested_map, ", ".join(available_map_names)))


def load_requested_world(client: carla.Client, requested_map: str) -> Tuple[carla.World, str]:
    ensure_map_name_exists(client, requested_map)

    world = client.get_world()
    current_map_name = _map_basename(world.get_map().name)
    if current_map_name == requested_map:
        return world, current_map_name

    world = client.load_world(requested_map)
    return world, requested_map
