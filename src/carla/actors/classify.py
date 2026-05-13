from typing import Optional

import carla


def find_hero_vehicle(world: carla.World) -> Optional[carla.Actor]:
    for vehicle in world.get_actors().filter("vehicle.*"):
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return None


def classify_vehicle_actor(actor: carla.Actor) -> str:
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
