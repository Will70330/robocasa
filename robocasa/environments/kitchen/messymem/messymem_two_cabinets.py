"""
messymem_two_cabinets.py — MessyMem two-cabinet scene task.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61).

Cabinet contents (both start CLOSED at reset):
  cab_1 — pantry items : ketchup, cereal, spaghetti_box, mayonnaise
  cab_2 — fruits       : apple, lime, lemon, banana, pear

Counter objects:
  Below cab_1 : orange, onion
  Below cab_2 : onion, bagged_food, pineapple

High-level goal: "Open the upper cabinets and find the banana."

Success (robocasa-native): cab_2 is open.
Primitive sequencing and subtask tracking live entirely in the external
planner (MockPlan in robocasa_sim.py, PlannerServer in Objective 4).
"""

import numpy as np

import robocasa.utils.env_utils as EnvUtils
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
    _CAB1_OBJECTS = ["ketchup", "cereal", "spaghetti_box", "mayonnaise"]
    _CAB2_OBJECTS = ["apple", "mustard", "lemon", "banana", "pear"]

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

    # ── Spawn offset ────────────────────────────────────────────────────────
    # Pull the robot spawn 0.15m back from the counter so the base clears
    # the cabinet when moving.
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
        name = self.target_obj.replace("_", " ")
        ep_meta["lang"] = f"Open the upper cabinets and find the {name}."
        ep_meta["target_obj"] = self.target_obj
        ep_meta["target_cab"] = self.target_cab_attr
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1: orange, onion ────────────────────────────────
        # (onion chosen over baguette because SAM3 reliably mislabels a
        # baguette as a "banana" from wrist-camera angles, polluting the
        # SG with a ghost banana_* node that breaks find-the-banana trials.)
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

        # ── Inside cab_1: pantry items — ketchup is the pinned pick target ───
        # Ketchup is locked to objaverse/ketchup/ketchup_7 — the thinnest
        # round bottle (4.4 × 4.4 × 16.5 cm), well within the Panda
        # gripper's ~8 cm opening. Pinned to a tight front-right region
        # of the shelf so PickFromCabinet's 30 cm pullback has clear
        # access even when only the right-hand door is open.
        #
        # The three distractors (cereal, spaghetti_box, mayonnaise)
        # are pinned to the back of the shelf so they stay clear of
        # the ketchup's pullback path — the mayonnaise jar was
        # toppling into the ketchup pull zone under the previous
        # free placement, and the analyzer was (incorrectly) marking
        # the pick as a failure on the collateral. Required for tasks
        # that probe whether the planner uses SG/KF memory to navigate
        # straight to the right cabinet for each target — without
        # distractors the planner could "free-win" cab_1 by always
        # opening it (only one object inside).
        cfgs.append(dict(
            name="ketchup",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/ketchup/ketchup_7/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab1,
                # Tight pin matching the clutter-chain's reliable
                # easy-pick pattern: 10×20 cm sampling region pulled
                # to the FRONT of the cabinet (near the door,
                # y=−0.78) so the pnp primitive's calibrated reach
                # lands on the ketchup. The previous wider region
                # (size=(0.4, 0.15), pos=(0, 0)) let the ketchup
                # spawn anywhere in a 40 cm × 15 cm zone in the
                # cabinet centre, and the pnp primitive's hardcoded
                # pre-grasp pose missed by 10+ cm on lateral-offset
                # spawns; the planner saw "missed entirely" / "closed
                # in the gap between ketchup and spaghetti box"
                # failures and gave up after retries on the
                # eval_config_robocasa_memory_chain pick task. The
                # clutter chain's tighter pin (size=(0.10, 0.20),
                # pos=(−0.75, −0.78)) didn't have this problem —
                # mirroring it here. x=0 (centred) because cab_1
                # has no side clutter on the same shelf; distractors
                # sit behind at y=+0.5.
                #
                # x shifted from 0.0 → -0.75 to match the clutter_chain
                # env's ketchup_1 pin, which reliably picks under the
                # same pnp primitive. The y-axis sampling window
                # (size_y=0.20) is wide enough that the bottle can land
                # 12cm deep in the cabinet on adverse seeds and time
                # out the arm-reach stages (seed-43 case); pushing x
                # to the side puts the bottle in the kinematically
                # reliable corner that the clutter_chain proved out.
                size=(0.10, 0.20),
                pos=(-0.75, -0.78),
            ),
        ))
        cab1_distractors = [
            ("cereal",        "cereal",        True),
            ("spaghetti_box", "spaghetti_box", False),  # non-graspable in registry
            ("mayonnaise",    "mayonnaise",    True),
        ]
        for name, group, graspable in cab1_distractors:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab1,
                    size=(0.8, 0.3),     # wide x, narrow y band
                    pos=(0.0, 0.5),      # back of shelf — y=+0.5 is
                                          # deeper into the cabinet
                                          # than the ketchup's y=0 pin
                                          # (see ketchup comment above
                                          # for the y-axis convention),
                                          # so distractors sit BEHIND
                                          # the ketchup and out of its
                                          # 30 cm pullback path.
                ),
            ))

        # ── Inside cab_2: mug is the pinned pick target; banana stays
        # for find_banana (Inspect/Open); other fruits are distractors.
        # Lime removed in favour of the mug — same role (graspable item
        # in cab_2 for the pick task), but pinned to the easy-access
        # front-right zone (mirroring ketchup's pin in cab_1) so
        # PickFromCabinetAndPlaceOnCounter has reliable kinematics.
        # Banana moves to back-right so it doesn't collide with the
        # mug at the front; find_banana only needs it visible, not
        # at the easy-pick spot.
        cab2_distractors = [
            ("apple",   "apple",   True),
            ("lemon",   "lemon",   True),
            ("pear",    "pear",    True),
        ]
        for name, group, graspable in cab2_distractors:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab2,
                    size=(0.6, 0.4),   # right-half, back-area only
                    pos=(0.4, -0.4),
                ),
            ))
        # Pinned mustard — Mustard001 from lightwheel: a thin yellow
        # squeeze bottle similar in shape to ketchup_7, so the same
        # pnp primitive that reliably picks ketchup in cab_1 also
        # reliably picks the mustard in cab_2. Previously this was a
        # mug (objaverse/mug/mug_9), which the pnp primitive could
        # grasp but whose handle / round profile occasionally caused
        # slips. Mustard's thin profile (~4 cm wide) is gripper-
        # friendly and matches the ketchup geometry. Pinned to the
        # same easy-pick zone the ketchup uses in cab_1 (tight 10×20
        # cm region, pulled forward to y=−0.78) so the pnp primitive's
        # calibrated reach lands on the bottle.
        cfgs.append(dict(
            name="mustard",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/lightwheel/mustard/Mustard003/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                # Mirror the ketchup pin in cab_1 — x=-0.75 lateral
                # offset matches the clutter_chain env's reliable
                # easy-pick position. See ketchup placement above.
                size=(0.10, 0.20),
                pos=(-0.75, -0.78),
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
                fixture=self.cab2,
                size=(0.4, 0.3),
                pos=(0.4, -0.4),     # back-right; visible for find_banana
                                       # but out of the mug's pick zone
            ),
        ))

        return cfgs

    # ── Chain-followup hooks ──────────────────────────────────────────

    def reset_chain_followup_neutral_pose(self, env=None):
        """Re-close both cabinet doors and teleport the robot to a
        NEUTRAL pose: centered between cab_1 and cab_2 with the base
        rotated 180° from spawn so the front camera faces away from
        the kitchen counter. Designed for chains where the test must
        force the planner to rely on memory rather than coincidentally
        seeing the relocated item in the live cam view.

        Does NOT teleport movable objects — items previously placed on
        the counter stay there.

        *env* is accepted for harness symmetry with other env_method
        hooks but unused.
        """
        # 1. Close both cabinet doors.
        try:
            self.cab1.close_door(env=self)
            self.cab2.close_door(env=self)
        except Exception as e:
            print(f"[MessymemTwoCabinets] close_door during neutral-pose "
                  f"followup failed: {e!r}")
        # 2. Centered base position — midway between cab_1 (x=1.75) and
        # cab_2 (x=4.55), at the spawn pullback distance from the
        # counter. Reuses the y/z components of the spawn anchor so the
        # pose stays inside the navigable region.
        try:
            spawn = self.init_robot_base_pos_anchor
            target_xy = np.array([3.15, spawn[1], spawn[2]])
            EnvUtils.set_robot_to_position(self, target_xy)
        except Exception as e:
            print(f"[MessymemTwoCabinets] robot teleport during "
                  f"neutral-pose followup failed: {e!r}")
        # 3. Rotate base yaw 180° so the front camera looks away from
        # the counter. Joint yaw is RELATIVE to init_robot_base_ori_anchor,
        # so qpos=np.pi means "spun half-turn from the spawn-facing
        # direction." With the spawn facing cab_1 / cab_2 (north toward
        # the cabinets), this points the agentview / wrist cameras
        # toward the south wall.
        try:
            self.sim.data.qpos[
                self.sim.model.get_joint_qpos_addr(
                    "mobilebase0_joint_mobile_yaw")
            ] = np.pi
        except Exception as e:
            print(f"[MessymemTwoCabinets] yaw rotation during "
                  f"neutral-pose followup failed: {e!r}")
        self.sim.forward()
        # 4. Settle a few frames so any contact resolves before
        # perception takes its next tick.
        try:
            zero = np.zeros(self.action_spec[0].shape)
            for _ in range(5):
                self.step(zero)
        except Exception:
            pass
        print("[MessymemTwoCabinets] chain followup: both cabinets "
              "re-closed; robot teleported to centered neutral pose "
              "rotated 180° from spawn (facing away from the counter).")

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
