"""
messymem_two_cabinets.py — MessyMem two-cabinet scene task.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61).

Cabinet contents (both start CLOSED at reset):
  cab_1 — pantry items : ketchup, cereal, spaghetti_box, canned_food, mayonnaise
  cab_2 — fruits       : apple, avocado, lime, lemon, banana, pear

Counter objects:
  Below cab_1 : orange, baguette
  Below cab_2 : onion, bagged_food, pineapple

High-level goal: "Open the upper cabinets and find the banana."

Success (robocasa-native): cab_2 is open.
Primitive sequencing and subtask tracking live entirely in the external
planner (MockPlan in robocasa_sim.py, PlannerServer in Objective 4).
"""

from robocasa.environments.kitchen.kitchen import *


class MessymemTwoCabinets(Kitchen):
    """
    MessyMem Two-Cabinet scene task.

    Sets up the messymem_001 kitchen with objects in two upper cabinets and
    on the counter below them.  Both cabinets start CLOSED.

    _check_success() follows the robocasa convention: it checks only the
    final physical end-state (cab_2 is open), not intermediate steps.
    Primitive sequencing is handled externally by the planner.
    """

    def __init__(self, *args, **kwargs):
        # Include "aigen" registry so aigen-only categories (spaghetti_box)
        # are resolvable.  The default ("objaverse", "lightwheel") would
        # produce an empty valid-category list and crash on rng.choice([]).
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    # Objects in each cabinet — used for random target selection.
    _CAB1_OBJECTS = ["ketchup", "cereal", "spaghetti_box", "canned_food", "mayonnaise"]
    _CAB2_OBJECTS = ["apple", "avocado", "lime", "lemon", "banana", "pear"]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )

        # Pick a random target object from either cabinet each episode.
        all_targets = (
            [(obj, "cab1") for obj in self._CAB1_OBJECTS] +
            [(obj, "cab2") for obj in self._CAB2_OBJECTS]
        )
        idx = self.rng.integers(len(all_targets))
        self.target_obj, self.target_cab_attr = all_targets[idx]

        # Robot spawns facing cab_1 (leftmost upper cabinet).
        self.init_robot_base_ref = self.cab1

    # ── Scene initialisation ──────────────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        name = self.target_obj.replace("_", " ")
        ep_meta["lang"] = f"Open the upper cabinets and find the {name}."
        ep_meta["target_obj"] = self.target_obj
        ep_meta["target_cab"] = self.target_cab_attr
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1: orange, baguette ─────────────────────────────
        cfgs.append(dict(
            name="orange",
            obj_groups="orange",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.cab1),
                size=(0.50, 0.20),
                pos=("ref", -1.0),
            ),
        ))
        cfgs.append(dict(
            name="baguette",
            obj_groups="baguette",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                reuse_region_from="orange",
                size=(1.0, 0.30),
                pos=(0, 0),
            ),
        ))

        # ── Counter below cab_2: onion, bagged_food, pineapple ────────────────
        cfgs.append(dict(
            name="onion",
            obj_groups="onion",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.cab2),
                size=(0.50, 0.20),
                pos=("ref", -1.0),
            ),
        ))
        cfgs.append(dict(
            name="bagged_food",
            obj_groups="bagged_food",
            graspable=False,
            placement=dict(
                fixture=self.counter,
                reuse_region_from="onion",
                size=(1.0, 0.30),
                pos=(0, 0),
            ),
        ))
        cfgs.append(dict(
            name="pineapple",
            obj_groups="pineapple",
            graspable=False,
            placement=dict(
                fixture=self.counter,
                reuse_region_from="onion",
                size=(1.0, 0.30),
                pos=(0, 0),
            ),
        ))

        # ── Inside cab_1: pantry items ────────────────────────────────────────
        cab1_items = [
            ("ketchup",       "ketchup",       True),
            ("cereal",        "cereal",        True),
            ("spaghetti_box", "spaghetti_box", False),  # non-graspable in registry
            ("canned_food",   "canned_food",   True),
            ("mayonnaise",    "mayonnaise",    True),
        ]
        for name, group, graspable in cab1_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab1,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        # ── Inside cab_2: fruits — banana is the high-level goal object ───────
        cab2_items = [
            ("apple",   "apple",   True),
            ("avocado", "avocado", True),
            ("lime",    "lime",    True),
            ("lemon",   "lemon",   True),
            ("banana",  "banana",  True),
            ("pear",    "pear",    True),
        ]
        for name, group, graspable in cab2_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab2,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        return cfgs

    # ── Success condition ─────────────────────────────────────────────────────

    # How open the target cabinet must be to count as "found" (0-1 normalized).
    # Default robocasa threshold is 0.90 (fully open); 0.35 lets the door be
    # ~1/3 open which is enough for perception to see inside.
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        """
        Final physical end-state: the cabinet containing the target object is
        open at least _OPEN_THRESHOLD of its full range.
        Follows the robocasa convention — one check, final state only.
        """
        target_cab = getattr(self, self.target_cab_attr)
        return target_cab.is_open(env=self, th=self._OPEN_THRESHOLD)
