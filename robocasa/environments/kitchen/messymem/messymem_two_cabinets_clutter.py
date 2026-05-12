"""
messymem_two_cabinets_clutter.py — MessyMem clutter-choice scene task.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61).

Both upper cabinets contain a mayonnaise bottle, but with very different
clutter levels:
  cab_1 (sparse)    : mayonnaise_1, apple                          — 2 items
  cab_2 (cluttered) : mayonnaise_2, ketchup, cereal, spaghetti_box,
                       canned_food, banana, lemon                   — 7 items

Counter objects: same as MessymemTwoCabinets to keep the surrounding
scene consistent with the existing task.

Use case: paired with `--with_exploration` so both cabinets are pre-opened
(populates scene graph contents and keyframe memory). The trial then asks
the planner to "find a mayonnaise bottle and navigate to the cabinet from
which it would be easier to retrieve one." The expected behaviour is that
the planner picks cab_1 (less cluttered → easier retrieval).

This is the comparison test for scene_graph+kf vs. scene_graph_only:
both modes see contents lists, but only scene_graph+kf sees keyframe
images that show the spatial clutter directly.
"""

from robocasa.environments.kitchen.kitchen import *


class MessymemTwoCabinetsClutterChoice(Kitchen):
    """
    MessyMem Two-Cabinet Clutter-Choice scene task.

    Both cab_1 and cab_2 contain a mayonnaise bottle, but cab_1 has only
    one other object while cab_2 is densely populated. Designed for the
    "given pre-explored state, can the planner pick the easier option?"
    test described in eval_config_robocasa_clutter.yaml.

    Both cabinets start CLOSED. Exploration that pre-opens them is the
    job of the eval harness (`--with_exploration`).

    _check_success() follows the robocasa convention: it checks only the
    final physical end-state (cab_1 is open — the easier-retrieval
    choice), not intermediate steps.
    """

    def __init__(self, *args, **kwargs):
        # Include "aigen" registry so aigen-only categories (spaghetti_box)
        # are resolvable.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    # Cabinet contents — mayonnaise lives in BOTH; the rest controls clutter.
    _CAB1_OBJECTS = ["mayonnaise_1", "apple"]
    _CAB2_OBJECTS = [
        "mayonnaise_2", "ketchup", "cereal", "spaghetti_box",
        "canned_food", "banana", "lemon",
    ]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )

        # Robot spawns facing cab_1 (leftmost upper cabinet).
        self.init_robot_base_ref = self.cab1

    # ── Spawn offset ────────────────────────────────────────────────────────
    # Pull the robot spawn 0.15m back from the counter so the base clears
    # the cabinet when moving (matches MessymemTwoCabinets).
    _SPAWN_PULLBACK = 0.15

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        import numpy as np
        yaw = self.init_robot_base_ori_anchor[2]
        self.init_robot_base_pos_anchor[:2] -= self._SPAWN_PULLBACK * np.array(
            [np.cos(yaw), np.sin(yaw)]
        )

    # ── Scene initialisation ──────────────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Find a mayonnaise bottle and navigate to the cabinet from "
            "which it would be easier to retrieve one."
        )
        # The expected target cabinet is cab_1 (sparser → easier retrieval).
        ep_meta["target_cab"] = "cab1"
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1: orange, onion ────────────────────────────────
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
            name="cab1_counter_onion",
            obj_groups="onion",
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

        # ── Inside cab_1: SPARSE — mayonnaise + apple ─────────────────────────
        cab1_items = [
            ("mayonnaise_1", "mayonnaise", True),
            ("apple",        "apple",      True),
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

        # ── Inside cab_2: CLUTTERED — mayonnaise + 6 others ───────────────────
        cab2_items = [
            ("mayonnaise_2", "mayonnaise",    True),
            ("ketchup",      "ketchup",       True),
            ("cereal",       "cereal",        True),
            ("spaghetti_box","spaghetti_box", False),  # non-graspable in registry
            ("canned_food",  "canned_food",   True),
            ("banana",       "banana",        True),
            ("lemon",        "lemon",         True),
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
    # 0.70 lets the door be ~30% open which is enough for perception.
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        """
        Final physical end-state: cab_1 (the sparser, easier-retrieval
        cabinet) is open. The eval harness's success_check is the
        authoritative judgement; this is just for robocasa-native
        reporting.
        """
        return self.cab1.is_open(env=self, th=self._OPEN_THRESHOLD)
