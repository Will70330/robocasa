"""
messymem_many_cabinets.py — MessyMem long-horizon keyframe-stress task.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61) — same shell
as the existing two-cabinets tasks, no new layout introduced.

Purpose: stress-test the keyframe-only ablations. The two-cabinets task only
opens 2 storages, so even a poor retriever has a reasonable chance of pulling
the right frame. This task lifts that to 4 distinct openable hinge cabinets,
all on the main wall, which inflates the keyframe-store size and the
analyzer's contents history. The planner must then recall which storage
holds the apple. Scene-graph modes read it directly from `cab_2.contents`;
keyframe modes have to retrieve the relevant frame out of a much larger
pool.

Storages (all start CLOSED at reset; --with_exploration in the eval harness
opens them all sequentially before the trial goal is submitted):

  cab_1   — upper-left hinge cabinet   : ketchup, cereal, mayonnaise
  cab_2   — upper-right hinge cabinet  : APPLE, banana, lemon   ← target
  stack_3 — lower-row hinge cabinet (L): paprika, cinnamon, turmeric
  stack_4 — lower-row hinge cabinet (R): condiment_bottle, jar, canned_food

Why stack_3/stack_4 instead of the wall-stack `top` / `fridge_top` upper
cabinets above the microwave/fridge? Those sit above z=2.0m, which is
above robocasa's default reset_region z_range (0.45, 1.50). Object
placement inside them fails at scene-build time. The lower hinge
cabinets at the bottom of the wall-mounted stacks (at z≈0.7m) are
within range and behave like normal openable hinge cabinets under
--auto_cabinets.

Why no fridge / dishwasher? Earlier drafts included both. Dropped because
the proportional `move_base_to` controller has no obstacle avoidance:
the dishwasher sits south of the kitchen island and getting back to the
main wall after parking there is unreliable; the fridge sits at the far
west of the main wall (x ≈ 1.0m) and the long traverse from cab_2 (x ≈
5.5m) sometimes hangs against counter geometry. Sticking to the four
hinge cabinets keeps the robot's path inside x ∈ [2.75, 5.55] on the
main wall — short hops, no obstacles. The 4-cabinet pool is still 2× the
two-cabinet baseline (and the analyzer captures multiple frames per
open, so the keyframe pool grows ~2.5×).

Counter objects (kept identical to MessymemTwoCabinets so the surrounding
scene matches the rest of the messymem benchmark): orange, baguette below
cab_1; onion, bagged_food, pineapple below cab_2.

High-level goal: "Find the apple."

Success (eval-harness driven): planner's final action targets cab_2 AND
planner self-terminates with empty plan. We DON'T check `cab_2.is_open` —
exploration already opened every storage, so that's trivially true and
would mask whether the planner actually identified cab_2 as the apple's
location. See eval_config_robocasa_many_cabinets.yaml for the check spec.
"""

from robocasa.environments.kitchen.kitchen import *


class MessymemManyCabinets(Kitchen):
    """
    MessyMem four-cabinet scene task — same kitchen as MessymemTwoCabinets,
    but with two additional openable hinge cabinets on the main wall to
    stress the keyframe-retrieval ablations.

    All storages start CLOSED. Pre-trial exploration is the eval harness's
    job (--with_exploration with the explicit action sequence in the
    eval YAML).

    _check_success() is a permissive sanity check (cab_2 is open). The
    authoritative judgement is the eval-harness success_check, which uses
    `planner_final_action_targets_fixture: cab_2` to verify the planner
    actually identified cab_2 as the apple's location.
    """

    # Cabinet contents — apple is the unique high-value target. Each storage
    # has visually distinguishable items so the analyzer can produce useful
    # contents lists in scene-graph mode. Categories are taken from the set
    # already proven to resolve in MessymemTwoCabinets (ketchup, cereal,
    # mayonnaise, apple, banana, lemon, canned_food, spaghetti_box) plus
    # the layout-decoration set (paprika, cinnamon, turmeric, condiment_bottle,
    # jar) which is known to load on this layout.
    _CAB1_OBJECTS    = ["ketchup", "cereal", "mayonnaise"]
    _CAB2_OBJECTS    = ["apple", "banana", "lemon"]                # apple is the target
    _STACK3_OBJECTS  = ["paprika", "cinnamon", "turmeric"]         # spices
    _STACK4_OBJECTS  = ["condiment_bottle", "jar", "canned_food"]  # jars / condiments

    # Contents lists exposed for the eval harness's ground_truth_contents
    # cross-reference. Mirrors the field used by the two-cabinets configs.
    _STORAGES = {
        "cab_1":   _CAB1_OBJECTS,
        "cab_2":   _CAB2_OBJECTS,
        "stack_3": _STACK3_OBJECTS,
        "stack_4": _STACK4_OBJECTS,
    }

    def __init__(self, *args, **kwargs):
        # "aigen" registry is required for spaghetti_box (objaverse + lightwheel
        # alone don't expose it). Matches MessymemTwoCabinets.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1       = self.register_fixture_ref("cab1",       dict(id="cab_1"))
        self.cab2       = self.register_fixture_ref("cab2",       dict(id="cab_2"))
        # stack_3 / stack_4 are bottom-row stacks whose level-0 fixture is a
        # HingeCabinet at z≈0.36m. The actual fixture name in the scene is
        # "<stack_name>_main_group_1" — pin to the full name with
        # full_name_check=True. We also assign each to the bare attribute
        # `self.stack_3` / `self.stack_4` so the policy server's
        # _resolve_fixture (which tries `getattr(base, name)` first) finds
        # them when the planner emits "OpenCabinet:stack_3".
        self.stack_3 = self.register_fixture_ref(
            "stack_3",
            dict(id="stack_3_main_group_1", full_name_check=True))
        self.stack_4 = self.register_fixture_ref(
            "stack_4",
            dict(id="stack_4_main_group_1", full_name_check=True))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1))

        # Robot spawns facing cab_1 (leftmost upper cabinet) — same as
        # MessymemTwoCabinets so the existing exploration / inspect arm
        # poses still apply.
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
        for cab in (self.cab1, self.cab2, self.stack_3, self.stack_4):
            cab.close_door(env=self)

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Find the apple."
        ep_meta["target_obj"] = "apple"
        ep_meta["target_cab"] = "cab2"
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1: orange, baguette ─────────────────────────────
        # (kept identical to MessymemTwoCabinets so the visual context
        # surrounding cab_1 is unchanged from the rest of the benchmark)
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
        for name in self._CAB1_OBJECTS:
            cfgs.append(dict(
                name=name,
                obj_groups=name,
                graspable=(name != "spaghetti_box"),
                placement=dict(
                    fixture=self.cab1,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        # ── Inside cab_2: fruits — APPLE is the target ────────────────────────
        for name in self._CAB2_OBJECTS:
            cfgs.append(dict(
                name=name,
                obj_groups=name,
                graspable=True,
                placement=dict(
                    fixture=self.cab2,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        # ── Inside stack_3 / stack_4 (lower-row hinge cabinets) ───────────────
        # These sit at z≈0.36m (cabinet center), so the level-0 shelf top is
        # well below 0.45m world-space — robocasa's default get_reset_regions
        # z_range=(0.45, 1.50) filters them out. Override the z_range via
        # sample_region_kwargs so the shelves are visible to the placement
        # sampler. The robot can still reach them; under --auto_cabinets the
        # door just teleports open regardless of arm pose.
        _LOW_Z_RANGE = (0.0, 1.50)
        for name in self._STACK3_OBJECTS:
            cfgs.append(dict(
                name=name,
                obj_groups=name,
                graspable=True,
                placement=dict(
                    fixture=self.stack_3,
                    sample_region_kwargs=dict(z_range=_LOW_Z_RANGE),
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))
        for name in self._STACK4_OBJECTS:
            cfgs.append(dict(
                name=name,
                obj_groups=name,
                graspable=True,
                placement=dict(
                    fixture=self.stack_4,
                    sample_region_kwargs=dict(z_range=_LOW_Z_RANGE),
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        return cfgs

    # ── Success condition ─────────────────────────────────────────────────────
    # Permissive env-side check; the eval harness's success_check is
    # authoritative. Returns True iff cab_2 is open at trial end.
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        return self.cab2.is_open(env=self, th=self._OPEN_THRESHOLD)
