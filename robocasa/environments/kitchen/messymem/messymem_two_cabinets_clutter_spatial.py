"""
messymem_two_cabinets_clutter_spatial.py — MessyMem spatial-clutter task.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61).

This task ACTIVELY TRAPS count-based reasoning. The "harder" cabinet
(cab_2, where the mayo is penned in by neighbours) deliberately has
FEWER items in its contents list than the "easier" cabinet (cab_1,
where the mayo is alone on the far side). A planner that judges
"easier" by number-of-items in the contents list will pick the WRONG
cabinet. Only the keyframe images carry the spatial-accessibility
signal needed to pick correctly.

Cabinet contents:

  cab_1 (mayo accessible) : 6 items — mayonnaise + ketchup + cereal +
                            spaghetti_box + canned_food + banana.
                            Mayonnaise pinned to the far-LEFT of the
                            cabinet, alone; other 5 items packed into
                            the right half. Counts: 6 (looks busy).

  cab_2 (mayo penned in)  : 3 items — mayonnaise + ketchup + canned_food.
                            Mayonnaise pinned to the CENTER; the two
                            other items pressed immediately against
                            its left and right sides. Counts: 3 (looks
                            sparse) — but in fact the mayo is wedged.

Counter objects: identical to MessymemTwoCabinets / ClutterChoice to
keep the surrounding scene consistent.

Test design:
  - scene_graph_only mode sees cab_1.contents has 6 items and
    cab_2.contents has 3 items → count-based reasoning picks cab_2
    (WRONG — mayo is wedged in cab_2).
  - scene_graph+kf sees keyframe images that show mayo alone in cab_1
    vs. wedged in cab_2 → picks cab_1 (RIGHT).

This is the strongest isolation: scene_graph_only doesn't just guess,
it gets actively misled.
"""

from robocasa.environments.kitchen.kitchen import *


class MessymemTwoCabinetsClutterSpatial(Kitchen):
    """
    MessyMem Spatial-Clutter scene task.

    cab_1 has 6 items with the mayonnaise alone on the far-left.
    cab_2 has 3 items with the mayonnaise wedged between two
    neighbours. The count-based signal favours the WRONG cabinet
    (cab_2, fewer items); only the spatial signal — visible in
    keyframes — favours the RIGHT one (cab_1, mayo isolated).

    Both cabinets start CLOSED. Pre-trial exploration is the eval
    harness's job (--with_exploration).

    _check_success() follows the robocasa convention: it checks only the
    final physical end-state (cab_1 is open — the easier-retrieval
    choice), not intermediate steps. The eval harness's success_check
    is the authoritative judgement.
    """

    def __init__(self, *args, **kwargs):
        # Include "aigen" registry so spaghetti_box is resolvable.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────
    #
    # Asymmetric on purpose — cab_2 has FEWER items so a count-based
    # heuristic would pick it (wrong).
    _CAB1_OBJECTS = [
        "mayonnaise_1", "ketchup_1", "cereal_1", "spaghetti_box_1",
        "canned_food_1", "banana_1",
    ]
    _CAB2_OBJECTS = ["mayonnaise_2", "ketchup_2", "canned_food_2"]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )

        # Robot spawns facing cab_1 (matches MessymemTwoCabinets).
        self.init_robot_base_ref = self.cab1

    # ── Spawn offset ────────────────────────────────────────────────────────
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
            "which it would be easier to retrieve one without knocking "
            "other items."
        )
        ep_meta["target_cab"] = "cab1"  # cab_1 is the easier-retrieval choice
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────
    #
    # `pos=(x, y)` is in [-1, 1] and anchors the inner sampling region
    # within the cabinet's interior; `size=(w, h)` is the inner region's
    # extent. Smaller size + specific pos → tighter constraint on where
    # the object lands. Robocasa's UniformRandomSampler then samples
    # uniformly within that inner region — but the rectangle is small
    # enough that placement is effectively deterministic.

    # Per-item inner-region size for cluster items. Small enough that 5
    # items fit in half a cabinet; large enough that the placement
    # sampler can find a non-overlapping spot.
    _CLUSTER_SIZE = (0.15, 0.30)
    # Mayonnaise gets a slightly tighter region so we can pin it.
    _MAYO_SIZE = (0.12, 0.30)

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1: orange, onion ────────────────────────────────
        # (onion replaces the original baguette: SAM3 mislabels a baguette
        # as "banana" from wrist-camera angles.)
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

        # ── Inside cab_1: mayonnaise on LEFT, cluster on RIGHT ────────────────
        # Mayonnaise pinned to the far-left of the cabinet, alone.
        cfgs.append(dict(
            name="mayonnaise_1",
            obj_groups="mayonnaise",
            graspable=True,
            placement=dict(
                fixture=self.cab1,
                size=self._MAYO_SIZE,
                pos=(-0.90, 0.0),
            ),
        ))
        # 5 cluster items packed into the right half of cab_1.
        # All 5 share roughly the same x-range (right side) but different
        # specific positions to keep them from overlapping.
        cab1_cluster = [
            ("ketchup_1",       "ketchup",       True,  (0.30,  0.0)),
            ("canned_food_1",   "canned_food",   True,  (0.55,  0.0)),
            ("cereal_1",        "cereal",        True,  (0.85,  0.0)),
            ("spaghetti_box_1", "spaghetti_box", False, (0.45, -0.6)),
            ("banana_1",        "banana",        True,  (0.75,  0.6)),
        ]
        for name, group, graspable, pos in cab1_cluster:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab1,
                    size=self._CLUSTER_SIZE,
                    pos=pos,
                ),
            ))

        # ── Inside cab_2: ONLY 3 items, mayo wedged between two ───────────────
        # The whole point of this scene is that cab_2 has FEWER items in
        # its contents list (so count-based reasoning picks it) but the
        # mayo is wedged — keyframes show that retrieval is harder here.
        # Mayonnaise pinned to the cabinet center.
        cfgs.append(dict(
            name="mayonnaise_2",
            obj_groups="mayonnaise",
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=self._MAYO_SIZE,
                pos=(0.0, 0.0),
            ),
        ))
        # Two items pressed against the mayonnaise's left and right sides.
        # Keep them close (|x| ≈ 0.20) so they read as "wedging" the mayo.
        cab2_cluster = [
            ("ketchup_2",     "ketchup",     True, (-0.20, 0.0)),  # left of mayo
            ("canned_food_2", "canned_food", True, ( 0.20, 0.0)),  # right of mayo
        ]
        for name, group, graspable, pos in cab2_cluster:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab2,
                    size=self._CLUSTER_SIZE,
                    pos=pos,
                ),
            ))

        return cfgs

    # ── Success condition ─────────────────────────────────────────────────────
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        return self.cab1.is_open(env=self, th=self._OPEN_THRESHOLD)
