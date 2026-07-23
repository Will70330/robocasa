"""
messymem_drawers_4x2.py — MessyMem 4x2 drawer-bank task (layout 71).

Exercises the hardcoded PullDrawer primitive + SG-overlay Gemini pointing on
the 4x2 drawer rig (layout_messymem_drawers_4x2, ID 71): two vertical
4-drawer stacks side by side -> 8 drawers.

Real fixture names (FixtureStack children are group-suffixed and 1-indexed,
_1=bottom .. _4=top — verified against the built env, NOT the drawer_1_0
convention the older two-drawer task assumed):
  drawer_1_main_group_1..4  — left column  (bottom -> top)
  drawer_2_main_group_1..4  — right column (bottom -> top)

Target: the top-right drawer (drawer_2_main_group_4, = SG drawer_7 after the
--gt_fixtures seed) holds a lemon. (A small round object — the narrow 0.45 m
drawers can't fit an elongated banana.)
Goal:   "Open the top drawer on the right side of the drawer bank ..."
Success: the target drawer is open past _OPEN_THRESHOLD.

This is a deterministic PullDrawer smoke test: the goal names the target
drawer spatially, so the planner can map it to the right SG node and the
SG-overlay pointing localizes that drawer's handle — no drawer-interior
perception (peek) is required for success.
"""
import os

import numpy as np

import robocasa
from robocasa.environments.kitchen.kitchen import *


class MessymemDrawers4x2(Kitchen):
    """MessyMem 4x2 drawer bank — PullDrawer + SG-overlay pointing test."""

    # Real fixture names on layout 71 (see module docstring).
    _TARGET_DRAWER = "drawer_2_main_group_4"      # top-right, holds banana
    _DISTRACTOR_DRAWER = "drawer_1_main_group_1"  # bottom-left, holds apple
    _ALL_DRAWERS = [f"drawer_{c}_main_group_{i}"
                    for c in (1, 2) for i in (1, 2, 3, 4)]

    # Drawers slide straight out, so a lower open threshold than a hinge door.
    _OPEN_THRESHOLD = 0.35
    _SPAWN_PULLBACK = 0.15

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("obj_registries", ("objaverse",))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.target_drawer = self.register_fixture_ref(
            "target_drawer",
            dict(id=self._TARGET_DRAWER, full_name_check=True))
        self.distractor_drawer = self.register_fixture_ref(
            "distractor_drawer",
            dict(id=self._DISTRACTOR_DRAWER, full_name_check=True))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.target_drawer))
        # Robot spawns facing the drawer bank.
        self.init_robot_base_ref = self.target_drawer

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        yaw = self.init_robot_base_ori_anchor[2]
        self.init_robot_base_pos_anchor[:2] -= self._SPAWN_PULLBACK * np.array(
            [np.cos(yaw), np.sin(yaw)])

    # ── Scene init: all 8 drawers start closed ────────────────────────────
    def _setup_scene(self):
        super()._setup_scene()
        for name in self._ALL_DRAWERS:
            fx = self.get_fixture(name, full_name_check=True)
            if fx is not None:
                fx.close_door(env=self)

    # ── Language ──────────────────────────────────────────────────────────
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = ("Open the top drawer on the right side of the "
                           "drawer bank to find the lemon.")
        ep_meta["target_obj"] = "lemon"
        ep_meta["target_drawer"] = self._TARGET_DRAWER
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────
    # Small round object only. Two constraints on the narrow 0.45 m drawers:
    #   - elongated items (banana) don't fit the sampler;
    #   - the lower drawers report no suitable sample region, so we place the
    #     single target object in the top-right target drawer only.
    # The distractor_drawer ref is kept (used for the "empty drawer" contrast
    # and future multi-object placement once lower-drawer regions are tuned).
    def _get_obj_cfgs(self):
        # pos=(0, -0.4) places the lemon in the front portion of the drawer
        # that clears the counter lip when it opens ~36% (world y≈-0.61), but
        # NOT right at the front lip (pos=-1.0 → y≈-0.71), which the lip
        # occludes from the raised-torso peek's downward viewing angle. This
        # spot is both exposed AND clearly visible in the peek's wrist frame.
        return [dict(
            name="lemon",
            obj_groups="lemon",
            graspable=True,
            placement=dict(
                fixture=self.target_drawer,
                size=(0.15, 0.12),
                pos=(0.0, -0.4),
            ),
        )]

    # ── Success: target drawer open ───────────────────────────────────────
    def _check_success(self):
        return self.target_drawer.is_open(env=self, th=self._OPEN_THRESHOLD)
