"""
messymem_drawers_4x2_memory.py — MessyMem 4x2 drawer memory-chain task.

A two-instruction memory-persistence scenario on the 4x2 drawer rig
(MessymemDrawers4x2 sibling). Two objects live in two different reachable
TOP drawers, far apart (left vs right column):

  lemon → drawer_1_main_group_4  (top-LEFT column,  SG id drawer_3)
  apple → drawer_2_main_group_4  (top-RIGHT column, SG id drawer_7)

Both objects sit at the FRONT of their drawer (pos=(0,-0.4), world y≈-0.61)
so they clear the counter lip and are visible in the Peek's wrist view when
the drawer opens ~36%.

The robot spawns facing the LEFT (lemon) drawer, so a left→right search order
is natural: it opens the lemon drawer first (sees a lemon, not the apple),
then the apple drawer (finds the apple). The eval config
(eval_config_robocasa_drawers_4x2_memory.yaml) chains two instructions:

  1. find_apple — search the drawers; along the way the Peek on the lemon
     drawer writes {lemon} into the SG, and the apple is found in drawer_7.
  2. find_lemon — SAME env, SG/KF preserved. The lemon's location is already
     in memory (drawer_3), so the planner should go straight back to that
     drawer WITHOUT re-opening the apple drawer or re-searching.

Success is scored by the config's planner_* checks (this task class defines
no _check_success beyond the default).
"""
import os

import numpy as np

import robocasa
from robocasa.environments.kitchen.messymem.messymem_drawers_4x2 import (
    MessymemDrawers4x2,
)
from robocasa.environments.kitchen.kitchen import *


class MessymemDrawers4x2Memory(MessymemDrawers4x2):
    """4x2 drawer rig with a lemon and an apple in two different top drawers,
    for a find-apple → recall-lemon memory chain."""

    _LEMON_DRAWER = "drawer_1_main_group_4"   # top-left  (SG id drawer_3)
    _APPLE_DRAWER = "drawer_2_main_group_4"   # top-right (SG id drawer_7)

    # ── Fixture references ────────────────────────────────────────────────
    def _setup_kitchen_references(self):
        # Skip MessymemDrawers4x2's single-target refs; set up our own.
        Kitchen._setup_kitchen_references(self)
        self.lemon_drawer = self.register_fixture_ref(
            "lemon_drawer",
            dict(id=self._LEMON_DRAWER, full_name_check=True))
        self.apple_drawer = self.register_fixture_ref(
            "apple_drawer",
            dict(id=self._APPLE_DRAWER, full_name_check=True))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.lemon_drawer))
        # Aliases so the inherited _setup_scene / helpers still resolve.
        self.target_drawer = self.apple_drawer
        self.distractor_drawer = self.lemon_drawer
        # Start facing the LEFT (lemon) drawer → natural left→right search.
        self.init_robot_base_ref = self.lemon_drawer

    # ── Language ──────────────────────────────────────────────────────────
    def get_ep_meta(self):
        ep_meta = super(MessymemDrawers4x2, self).get_ep_meta()
        # Chain instruction text comes from the eval config; this is a
        # generic fallback describing the scene.
        ep_meta["lang"] = ("Find the requested fruit in the drawer bank; a "
                           "lemon and an apple are in two different drawers.")
        ep_meta["lemon_drawer"] = self._LEMON_DRAWER
        ep_meta["apple_drawer"] = self._APPLE_DRAWER
        return ep_meta

    # ── Object placements: lemon (top-left) + apple (top-right), both front ─
    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="lemon",
            obj_groups="lemon",
            graspable=True,
            placement=dict(fixture=self.lemon_drawer,
                           size=(0.15, 0.12), pos=(0.0, -0.4)),
        ))
        cfgs.append(dict(
            name="apple",
            obj_groups="apple",
            graspable=True,
            placement=dict(fixture=self.apple_drawer,
                           size=(0.15, 0.12), pos=(0.0, -0.4)),
        ))
        return cfgs
