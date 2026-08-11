"""
messymem_five_cabinets_kitchen.py — the stocked U-kitchen.

  MessymemFiveCabinetsKitchen   layout 75
  scene: configs/sim/scene_objects/five_cab_kitchen_v1.json

MessymemFiveCabinets with its 89 objects loaded from a layout-builder
session. Nothing is sampled: every object is pinned to the world pose it
was placed at, so the scene is byte-identical on every seed and every
reset, and the eval's ground truth is the session file itself.

Contents, by fixture:
  cab_001  produce      carrot, beet, asparagus, artichoke x2, corn x2, tomato
  cab_002  pantry       cereal x3, canned_food x11, jam x2, boxed_food x2,
                        flour_bag
  cab_003  dishes       plate x10, bowl x3, mug, glass_cup x7
  cab_004  condiments   MUSTARD, ketchup, mayonnaise, olive oil, syrup,
                        vinegar, paprika x2, pepper shaker, cinnamon, turmeric
  cab_005  proteins     chicken drumstick x4, fish x2, egg x6, kebabs x3
  counters             SAUSAGE + lobster/shrimp/lamb chop (front),
                        BREAD x3 + onion/pickle/tomato slice (back)

The hot-dog task's three targets are the sausage on the front counter, any
bread on the back counter, and the mustard inside cab_004 — one item per
wall, so the task cannot be solved without turning around.
"""
from robocasa.environments.kitchen.messymem.authored_scene import AuthoredScene
from robocasa.environments.kitchen.messymem.messymem_five_cabinets import (
    MessymemFiveCabinets,
)

# Named sub-regions of a fixture that get their own scene-graph node.
#
# counter_left is one 5 m run with the sink partway along it, so "the
# counter to the right of the sink" is not addressable as a fixture. Naming
# the stretch between the sink and the back wall makes it a first-class
# node the planner can route to and place on, straight from the
# instruction text — no backend special-case for where deliveries go.
#
# Ids are used verbatim as scene-graph node ids, so they follow the
# <name>_<n> convention the rest of the graph uses. Note that the keyframe
# retriever will not anchor on this node regardless: _ANCHOR_SKIP_BASES
# excludes every base starting with "counter", by design.
NAMED_REGIONS = {
    "counter_next_to_sink_0": dict(
        fixture="counter_left", label="counter next to sink",
        x=(-1.00, -0.35), y=(-3.30, 0.00), z=(0.90, 1.40),
        description="the clear stretch of counter between the sink and the "
                    "back wall",
    ),
}


class MessymemFiveCabinetsKitchen(AuthoredScene, MessymemFiveCabinets):
    _SESSION_FILE = "configs/sim/scene_objects/five_cab_kitchen_v1.json"

    # Categories a task may ask the arm to move. Plates, bowls and the
    # glassware in cab_003 are wider than the jaws; the produce and
    # proteins are there as scenery.
    _GRASPABLE = {"sausage", "bread_flat", "mustard", "ketchup", "mayonnaise",
                  "canned_food", "cereal", "jam", "boxed_food", "egg",
                  "tomato", "onion", "pickle", "carrot", "corn", "mug"}

    def _authored_entries(self):
        # The builder marks everything graspable; narrow that to the
        # categories the jaws can actually close on, so a task cannot
        # target a 0.3 m plate.
        entries = super()._authored_entries()
        for e in entries:
            e["graspable"] = (e.get("category") or
                              e["name"].rsplit("_", 1)[0]) in self._GRASPABLE
        return entries

    @property
    def named_regions(self):
        return {k: dict(v) for k, v in NAMED_REGIONS.items()}
