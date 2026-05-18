"""
messymem_two_drawers.py — MessyMem two-drawer scene task.

Sibling to messymem_two_cabinets.py — same "two distinct fixture targets far
apart on the main wall" story, but the targets are ground-level drawer
stacks instead of upper hinge cabinets so the OpenDrawer trained primitive
gets exercised end-to-end through the planner.

Scene: layout_messymem_drawers (layout ID 64), a minimal kitchen with two
drawer stacks ~3.6 m apart on the main wall and no other cabinets.

Stack layout (FixtureStack names child drawers <stack>_<idx>, 0=top):
  drawer_1_0 — top drawer of left stack  : pantry items
  drawer_1_1 — bottom drawer of left stack
  drawer_2_0 — top drawer of right stack : fruits incl. banana   ← target
  drawer_2_1 — bottom drawer of right stack

High-level goal: "Open the drawers and find the banana." (banana inside
drawer_2_0; planner picks which stack to inspect.)

Success: target drawer (the one holding the banana) is open past _OPEN_THRESHOLD.
"""
import os

import robocasa
from robocasa.environments.kitchen.kitchen import *


class MessymemTwoDrawers(Kitchen):
    """
    MessyMem two-drawer scene task — mirrors MessymemTwoCabinets but with
    drawer stacks instead of hinge cabinets. Both targets are the *top*
    drawers of their respective stacks (drawer_1_0 and drawer_2_0).

    Per-trial randomization: a target object is chosen at reset; the
    planner's task is to locate it via Inspect / OpenDrawer.
    """

    def __init__(self, *args, **kwargs):
        # aigen registry needed for spaghetti_box (matches two_cabinets).
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # Objects in each drawer — used for random target selection.
    _DRAWER1_OBJECTS = ["ketchup", "cereal", "spaghetti_box", "canned_food", "mayonnaise"]
    _DRAWER2_OBJECTS = ["apple", "lime", "lemon", "banana", "pear"]

    # ── Fixture references ────────────────────────────────────────────────────

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # FixtureStack creates child fixtures named `<stack>_<idx>`, so the
        # top drawer of stack `drawer_1` is `drawer_1_0`. We pass
        # full_name_check=True so the substring matcher doesn't return a
        # random child (drawer_1_0 vs drawer_1_1) — we want the *top* drawer
        # specifically.
        self.drawer1 = self.register_fixture_ref(
            "drawer1", dict(id="drawer_1_0", full_name_check=True)
        )
        self.drawer2 = self.register_fixture_ref(
            "drawer2", dict(id="drawer_2_0", full_name_check=True)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer1)
        )

        # Pick a random target object — banana is the canonical "find me"
        # goal but any of the items can be the trial target.
        all_targets = (
            [(obj, "drawer1") for obj in self._DRAWER1_OBJECTS] +
            [(obj, "drawer2") for obj in self._DRAWER2_OBJECTS]
        )
        idx = self.rng.integers(len(all_targets))
        self.target_obj, self.target_drawer_attr = all_targets[idx]

        # Robot spawns facing drawer_1 (left stack).
        self.init_robot_base_ref = self.drawer1

    # ── Spawn offset (matches two_cabinets) ───────────────────────────────────
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
        # All four drawers start closed.
        self.drawer1.close_door(env=self)
        self.drawer2.close_door(env=self)

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        name = self.target_obj.replace("_", " ")
        ep_meta["lang"] = f"Open the drawers and find the {name}."
        ep_meta["target_obj"] = self.target_obj
        ep_meta["target_drawer"] = self.target_drawer_attr
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Inside drawer_1: pantry items ─────────────────────────────────────
        drawer1_items = [
            ("ketchup",       "ketchup",       True),
            ("cereal",        "cereal",        True),
            ("spaghetti_box", "spaghetti_box", False),
            ("canned_food",   "canned_food",   True),
            ("mayonnaise",    "mayonnaise",    True),
        ]
        for name, group, graspable in drawer1_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.drawer1,
                    size=(0.6, 0.30),
                    pos=(None, None),
                ),
            ))

        # ── Inside drawer_2: fruits — banana is the high-level goal ───────────
        drawer2_other_items = [
            ("apple", "apple", True),
            ("lime",  "lime",  True),
            ("lemon", "lemon", True),
            ("pear",  "pear",  True),
        ]
        for name, group, graspable in drawer2_other_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.drawer2,
                    size=(0.6, 0.30),
                    pos=(None, None),
                ),
            ))
        cfgs.append(dict(
            name="banana",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/banana/banana_8/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.drawer2,
                size=(0.5, 0.20),
                pos=(0.4, 1.0),
            ),
        ))

        return cfgs

    # ── Success condition ─────────────────────────────────────────────────────

    # How open the target drawer must be (0-1 normalized). Drawers slide
    # straight out so a lower threshold is appropriate than for hinge doors.
    _OPEN_THRESHOLD = 0.50

    def _check_success(self):
        """Target drawer (the one holding the goal object) is open past threshold."""
        target_drawer = getattr(self, self.target_drawer_attr)
        return target_drawer.is_open(env=self, th=self._OPEN_THRESHOLD)
