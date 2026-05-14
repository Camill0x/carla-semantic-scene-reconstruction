import time
from typing import Optional, Tuple

import carla


def find_hero_vehicle(world: carla.World) -> Optional[carla.Actor]:
    """Return the hero vehicle actor from the current CARLA world, if present."""
    for vehicle in world.get_actors().filter("vehicle.*"):
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return None


def wait_for_hero_vehicle(
    client: carla.Client,
    *,
    timeout_s: float = 15.0,
    poll_interval_s: float = 0.1,
    require_sync: bool = False,
) -> Tuple[carla.World, carla.Actor]:
    """Wait until the current world exposes a hero vehicle and optionally synchronous mode."""
    deadline = time.monotonic() + timeout_s
    last_reason = "Hero vehicle not found."

    while True:
        world = client.get_world()
        settings = world.get_settings()
        hero = find_hero_vehicle(world)

        if hero is not None and (not require_sync or settings.synchronous_mode):
            return world, hero

        map_name = world.get_map().name
        if hero is None:
            last_reason = f"Hero vehicle not found in current world: {map_name}"
        else:
            last_reason = f"Current world is not yet running in synchronous mode: {map_name}"

        if time.monotonic() >= deadline:
            raise TimeoutError(last_reason)

        time.sleep(poll_interval_s)


def classify_vehicle_actor(actor: carla.Actor) -> str:
    """Map a CARLA vehicle actor to one of the project object classes."""
    type_id = actor.type_id.lower()

    if "bus" in type_id:
        return "bus"

    truck_keywords = [
        "truck",
        "firetruck",
        "carlacola",
        "european_hgv",
        "sprinter",
        "van",
        "ambulance",
    ]
    if any(k in type_id for k in truck_keywords):
        return "truck"

    motorcycle_keywords = [
        "motorcycle",
        "harley",
        "vespa",
        "yamaha",
        "kawasaki",
        "scooter",
    ]
    if any(k in type_id for k in motorcycle_keywords):
        return "motorcycle"

    bicycle_keywords = [
        "bicycle",
        "bike",
        "crossbike",
        "omafiets",
        "diamondback",
        "gazelle",
        "bh.",
    ]
    if any(k in type_id for k in bicycle_keywords):
        return "bicycle"

    wheels = actor.attributes.get("number_of_wheels")
    if wheels is not None:
        try:
            if int(wheels) == 2:
                return "motorcycle"
        except ValueError:
            pass

    return "car"
