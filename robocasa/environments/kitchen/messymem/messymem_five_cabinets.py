"""
messymem_five_cabinets.py — U-shaped kitchen scene for the ingredient-
gathering and kitchen-cleanup evaluations.

  MessymemFiveCabinets   layout 75   layout_messymem_five_cabinets

Five upper hinge cabinets, a sink counter and a fridge arranged in a U:

  cab_001 x=1.15  cab_002 x=2.75  cab_003 x=4.35   back wall  (y =  0)
  cab_004 x=1.95  cab_005 x=3.55  fridge  x=5.20   front wall (y = -5)
  sink    y=-3.75                                  left wall  (x = -1)

cab_004 / cab_005 sit on the midpoints of the back wall's gaps, so no
cabinet is ever directly opposite another — a wrist frame taken at one
station never has a second cabinet centred behind it.

Ids are zero-padded to three digits for the reason documented in
messymem_n_uppers.py: Kitchen.get_fixture matches by SUBSTRING, so an
unpadded ``cab_1`` would also match ``cab_10_main_group``.

The scene carries NO objects yet — ``_get_obj_cfgs`` returns []. Contents
come from the placement tool and land in ``_FIXED_CONTENTS``-style tables
here. Instruction text and success checks live in the eval config.
"""
from robocasa.environments.kitchen.kitchen import *


class MessymemFiveCabinets(Kitchen):
    """U-shaped 5-cabinet kitchen. Scene only — no objects, no goal."""

    _CAB_IDS = ("cab_001", "cab_002", "cab_003", "cab_004", "cab_005")

    def __init__(self, *args, **kwargs):
        # mustard / mayonnaise are lightwheel-only and pineapple aigen-only;
        # register all three now so object tables added later resolve.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # Register under the ref name AND as a bare attribute: the policy
        # server's _resolve_fixture tries getattr(base, "cab_004") first
        # when the planner emits "OpenCabinet:cab_004".
        self.cabs = []
        for cab_id in self._CAB_IDS:
            cab = self.register_fixture_ref(cab_id, dict(id=cab_id))
            setattr(self, cab_id, cab)
            self.cabs.append(cab)
        for i, cab in enumerate(self.cabs, start=1):
            setattr(self, f"cab{i}", cab)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))

        # Three separate counters, so name them rather than using
        # FixtureType.COUNTER, which would resolve to whichever is nearest.
        self.counter_back = self.register_fixture_ref(
            "counter_back", dict(id="counter_back")
        )
        self.counter_front = self.register_fixture_ref(
            "counter_front", dict(id="counter_front")
        )
        self.counter_left = self.register_fixture_ref(
            "counter_left", dict(id="counter_left")
        )
        # Generic alias — the burger task's drop-off is the sink counter.
        self.counter = self.counter_left

        # Stock robocasa spawn: parked in front of cab_001, same as the
        # other messymem scenes.
        self.init_robot_base_ref = self.cabs[0]

    # ── Scene initialisation ──────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        for cab in self.cabs:
            cab.close_door(env=self)

    # ── Object placements ─────────────────────────────────────────────

    def _get_obj_cfgs(self):
        return []

    def _check_success(self):
        # Success is scored by the eval config's checks, not here.
        return False
