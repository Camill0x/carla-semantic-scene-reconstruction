from typing import List, Tuple

import carla


def _map_basename(map_path: str) -> str:
    """Return the basename of a CARLA map path."""
    return map_path.rsplit("/", 1)[-1]


def _available_map_names(client: carla.Client) -> List[str]:
    """Return the sorted set of available CARLA map names."""
    available_maps = client.get_available_maps()
    return sorted({_map_basename(map_path) for map_path in available_maps})


def ensure_map_name_exists(client: carla.Client, requested_map: str) -> None:
    """Validate that the requested CARLA map exists on the server."""
    available_map_names = _available_map_names(client)
    if requested_map in available_map_names:
        return

    raise ValueError("Map %r does not exist. Choose one of: %s" % (requested_map, ", ".join(available_map_names)))


def get_current_world(client: carla.Client) -> Tuple[carla.World, str]:
    """Return the current CARLA world together with its map name."""
    world = client.get_world()
    return world, _map_basename(world.get_map().name)


def load_requested_world(client: carla.Client, requested_map: str) -> Tuple[carla.World, str]:
    """Load the requested CARLA map if it is not already active."""
    ensure_map_name_exists(client, requested_map)

    world, current_map_name = get_current_world(client)
    if current_map_name == requested_map:
        return world, current_map_name

    world = client.load_world(requested_map)
    return world, requested_map
