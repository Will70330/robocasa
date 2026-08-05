"""
messymem_two_cabinets_clutter_chain.py — Clutter-spatial + memory-chain
variant.

Sibling of messymem_two_cabinets_clutter_spatial.py, redesigned for
chained-instruction memory evaluation. The cabinet contents are
arranged so that:

  - find_mustard (instruction 1) inspects both cabinets; the
    mustard lives in cab_2 alongside ketchup and cereal, all
    clustered together. The side effect is that the analyzer captures
    interior keyframes showing the spatial arrangement of items
    around the ketchup in BOTH cabinets.
  - pick_ketchup_careful (instruction 2) requires retrieving the
    ketchup. Ketchup is in both cabinets, but only cab_1's ketchup
    is accessible (alone on the far-left of a 6-item shelf). cab_2's
    ketchup is wedged in the middle of 3 items. The planner must
    read the spatial signal back from the find_mustard keyframes
    — contents lists alone do not distinguish "wedged" from "alone".

Cabinet contents:

  cab_1 (ketchup accessible) : 6 items — ketchup + lemon + apple +
                               mug + banana + pear. Ketchup pinned to
                               the far-LEFT, alone; the other 5 items
                               packed into the right half. Counts: 6
                               (looks busy) — but the ketchup is in
                               fact clear.

  cab_2 (ketchup wedged)     : 3 items — mustard + ketchup + cereal.
                               Ketchup pinned to the CENTER; mustard
                               and cereal pressed against its left
                               and right sides. Counts: 3 (looks
                               sparse) — but ketchup is in fact
                               wedged. This is also where the mustard
                               (instruction 1's target) lives.

This is the SAME anti-count trick as MessymemTwoCabinetsClutterSpatial:
the cluttered cabinet has FEWER items. A planner that picks the
"easier" cabinet by item-count gets misled toward cab_2 (wrong).

Chain-followup hook: between instructions the eval harness calls
reset_for_chain_followup to close both cabinet doors and return the
robot to its cab_1 spawn. With this content layout the spawn happens
to be at the right cabinet (cab_1 = accessible ketchup), so proximity
and memory align rather than fighting — the test still requires the
planner to consult find_mustard's keyframes to confirm cab_1 is
the less cluttered option, but it doesn't have to override a
proximity bias toward the wrong cabinet.
"""

import os

import numpy as np

import robocasa
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import *


class MessymemTwoCabinetsClutterChain(Kitchen):
    """
    MessyMem Clutter-Chain task — clutter-spatial layout with a
    chain-followup hook for memory-chain evaluations.

    cab_1: 6 items, ketchup alone on the far-left, 5 others (lemon,
    apple, mug, banana, pear) packed on the right.
    cab_2: 3 items, mustard + ketchup + cereal all clustered in
    the middle (also where instruction 1's mustard lives).

    Both cabinets start closed. reset_for_chain_followup re-closes
    them and teleports the robot back to the cab_1 spawn — which is
    also the accessible-ketchup cabinet, so spawn proximity aligns
    with the correct choice rather than fighting it.
    """

    def __init__(self, *args, **kwargs):
        # Include "aigen" registry so cereal / mustard variants
        # resolve (matches clutter_spatial).
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────
    _CAB1_OBJECTS = [
        "ketchup_1", "lemon_1", "apple_1", "mug_1", "banana_1", "pear_1",
    ]
    _CAB2_OBJECTS = ["mustard_2", "ketchup_2", "cereal_2"]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )

        # Robot spawns facing cab_1 — the WRONG cabinet (ketchup
        # wedged). Proximity bias is intentional: instruction 2 must
        # override it via memory-driven spatial reasoning.
        self.init_robot_base_ref = self.cab1

    # ── Spawn offset ─────────────────────────────────────────────────
    _SPAWN_PULLBACK = 0.15

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        yaw = self.init_robot_base_ori_anchor[2]
        self.init_robot_base_pos_anchor[:2] -= self._SPAWN_PULLBACK * np.array(
            [np.cos(yaw), np.sin(yaw)]
        )

    # ── Scene initialisation ─────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)

    # ── Language metadata ────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Find the mustard in the upper cabinets, then retrieve "
            "the ketchup from the cabinet where it can be picked up "
            "without disturbing other items."
        )
        # cab_1 is the easier-retrieval choice for ketchup.
        ep_meta["target_cab"] = "cab1"
        return ep_meta

    # ── Object placements ────────────────────────────────────────────

    # Per-item inner-region size for cluster items.
    _CLUSTER_SIZE = (0.15, 0.30)
    # Ketchup gets a slightly tighter region so we can pin it.
    _KETCHUP_SIZE = (0.12, 0.30)

    def _get_obj_cfgs(self):
        cfgs = []

        # ── Counter below cab_1 (mirrors clutter_spatial) ─────────────
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

        # ── Counter below cab_2 (mirrors clutter_spatial) ─────────────
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

        # ── Inside cab_1: 6 items, ketchup ALONE on the far-left ──────
        # Ketchup pinned to the far-left of the cabinet, alone. The
        # other 5 items packed into the right half. This is the
        # accessible-ketchup cabinet — the planner should pick from
        # here in instruction 2.
        cfgs.append(dict(
            name="ketchup_1",
            # Same ketchup_7 asset the memchain config pins for its
            # easy-pick zone — thin 4.4cm-wide bottle, gripper-friendly
            # for the diffusion OpenCabinet/PickFromCabinet policies.
            # Avoids the random-sampler picking a chunkier variant
            # that the pnp primitive's pre-grasp standoff can't
            # accommodate.
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/ketchup/ketchup_7/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab1,
                # Custom tight region: 0.10×0.20 sampling so the
                # bottle reliably lands within ~5mm of pos. With the
                # larger _KETCHUP_SIZE=(0.12, 0.30) the bottle could
                # jitter back 0.05m, putting it out of where the pnp
                # primitive's reach actually ends up.
                size=(0.10, 0.20),
                # x=-0.75, y=-0.78: pushed further left and forward
                # so the bottle lands where the wrist-cam shows the
                # gripper arriving. y=-0.78 is the closest we can get
                # to the door before the sampler runs out of bounds
                # (y_min - bbox half ≈ -0.99). x=-0.75 is still ~1.0
                # normalized units away from the right-half cluster.
                pos=(-0.75, -0.78),
            ),
        ))
        cab1_cluster = [
            ("lemon_1",  "lemon",  True, (0.30, 0.0)),   # front-left of right cluster
            ("apple_1",  "apple",  True, (0.55, 0.0)),
            ("mug_1",    "mug",    True, (0.85, 0.0)),
            ("banana_1", "banana", True, (0.45, -0.6)),
            ("pear_1",   "pear",   True, (0.75, 0.6)),
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

        # ── Inside cab_2: 3 items, ketchup WEDGED in the middle ───────
        # Mustard (instruction 1's target), ketchup, and cereal all
        # clustered together. Anti-count trick: cluttered cabinet has
        # FEWER items, so a count-based heuristic picks it (wrong).
        #
        # Positions and sizes match MessymemTwoCabinetsClutterSpatial's
        # working pattern (|x|=0.20, size=(0.15, 0.30) for wedging items,
        # size=(0.12, 0.30) for the centered item) — tighter spacings
        # caused load_model to exhaust 50 retries when worst-case bbox
        # samples overlapped.
        cfgs.append(dict(
            name="ketchup_2",
            obj_groups="ketchup",
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=self._KETCHUP_SIZE,
                pos=(0.0, 0.0),
            ),
        ))
        # Pin mustard to Mustard001 — iconic yellow bottle with a
        # large dark-blue "Mustard" label and burger illustration,
        # much more recognizable than the random mustard variant the
        # category sampler picks. Yellow body also contrasts cleanly
        # with the red ketchup so the analyzer / planner can easily
        # tell them apart.
        cfgs.append(dict(
            name="mustard_2",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/lightwheel/mustard/Mustard001/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=self._CLUSTER_SIZE,
                pos=(-0.20, 0.0),  # pressed against left side of ketchup
            ),
        ))
        cfgs.append(dict(
            name="cereal_2",
            obj_groups="cereal",
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=self._CLUSTER_SIZE,
                pos=(0.20, 0.0),  # pressed against right side of ketchup
            ),
        ))

        return cfgs

    # ── Chain-followup hook ──────────────────────────────────────────

    def reset_for_chain_followup(self, env=None):
        """Re-close both cabinet doors and return the robot to its
        cab_1 spawn pose. Called by the eval harness between chain
        members so instruction 2 always starts from the same
        physical state — robot proximity-biased toward cab_1 (the
        wrong choice) so success requires actively navigating to
        cab_2 via memory-driven spatial reasoning.

        Idempotent across repeated invocations within the same trial.
        Does NOT reset MuJoCo object positions — the chain's
        reset-suppression keeps the SG/KF memory and the physical
        item layout intact.

        *env* is accepted for harness symmetry with other env_method
        hooks but unused.
        """
        try:
            self.cab1.close_door(env=self)
            self.cab2.close_door(env=self)
        except Exception as e:
            print(f"[ClutterChain] close_door during followup "
                  f"failed: {e!r}")
        try:
            EnvUtils.set_robot_to_position(self, self.init_robot_base_pos)
        except Exception as e:
            print(f"[ClutterChain] robot teleport during followup "
                  f"failed: {e!r}")
        self.sim.forward()
        # Settle a few frames so any contact resolves before
        # perception takes its next tick.
        try:
            zero = np.zeros(self.action_spec[0].shape)
            for _ in range(5):
                self.step(zero)
        except Exception:
            pass
        print("[ClutterChain] chain followup: both cabinets re-closed, "
              "robot returned to cab_1 spawn.")

    def close_cabinets_for_chain_followup(self, env=None):
        """Re-close both cabinet doors but do NOT teleport the robot.
        The robot stays in whatever pose the prior chain member left
        it (typically in front of cab_2 after find_mustard, since the
        last action there was opening / inspecting cab_2).

        Designed for a stronger version of the clutter-spatial test
        where proximity bias now points at the WRONG cabinet (cab_2,
        the cluttered one) so a successful pick from cab_1 requires
        actively overriding proximity using the find_mustard keyframes.

        Idempotent. Does NOT reset MuJoCo object positions — the
        chain's reset-suppression keeps SG/KF memory and the physical
        item layout intact.

        *env* is accepted for harness symmetry with other env_method
        hooks but unused.
        """
        try:
            self.cab1.close_door(env=self)
            self.cab2.close_door(env=self)
        except Exception as e:
            print(f"[ClutterChain] close_door during followup "
                  f"failed: {e!r}")
        self.sim.forward()
        try:
            zero = np.zeros(self.action_spec[0].shape)
            for _ in range(5):
                self.step(zero)
        except Exception:
            pass
        print("[ClutterChain] chain followup: both cabinets re-closed; "
              "robot pose preserved from prior chain member.")

    # ── Success condition ────────────────────────────────────────────
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        # Final physical state: cab_1 (the easier-retrieval choice)
        # is open. The eval harness's success_check is the
        # authoritative judgement; this is a coarse sanity check.
        return self.cab1.is_open(env=self, th=self._OPEN_THRESHOLD)


class MessymemTwoCabinetsClutterChainSwapped(MessymemTwoCabinetsClutterChain):
    """Swapped-layout variant of MessymemTwoCabinetsClutterChain.

    Cabinet contents are mirrored: the 6-item set with ketchup-alone-
    on-the-far-left (the ACCESSIBLE choice) now lives in cab_2 (right
    cabinet); the 3-item cluster with ketchup-wedged-in-the-middle
    (the CLUTTERED choice, also containing the mustard target) now
    lives in cab_1 (left cabinet).

    Why: the original task has cab_1 as the correct cabinet, which
    coincides with the robot's spawn pose. After close_cabinets_for_
    chain_followup preserves the find_mustard exit pose (robot at the
    cluttered cabinet = cab_2 originally), proximity misleads. This
    swapped variant flips the layout so the cluttered cabinet (and
    therefore the find_mustard exit pose) is now cab_1, and the
    correct ketchup cabinet is cab_2. This controls against any
    planner-side or training-data bias toward "always cab_1" — if the
    pattern of preference is truly about consulting the find_mustard
    keyframes (rather than fixture identity), the planner should still
    pick the spacious cabinet, which is now cab_2.

    Implementation: just swaps `self.cab1` and `self.cab2` before
    calling the parent's _get_obj_cfgs. Every cfg that referenced
    self.cab1 now lands its items in the actual cab_2 fixture and vice
    versa. Counter items below the cabinets also swap, which is
    harmless background dressing. Restores the references afterward
    so the rest of the env (success checks, reset hooks, etc.) still
    sees cab_1/cab_2 in their canonical positions.
    """

    def _get_obj_cfgs(self):
        self.cab1, self.cab2 = self.cab2, self.cab1
        try:
            return super()._get_obj_cfgs()
        finally:
            self.cab1, self.cab2 = self.cab2, self.cab1

    def _check_success(self):
        # Mirror the parent's coarse check, but the easier-retrieval
        # cabinet is now cab_2 (the spacious one) in this variant.
        return self.cab2.is_open(env=self, th=self._OPEN_THRESHOLD)
