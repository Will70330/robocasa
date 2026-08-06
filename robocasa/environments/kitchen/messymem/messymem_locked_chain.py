"""
messymem_locked_chain.py — three-cabinet locked-left variant for
memory-chain evaluations.

Designed for layout_messymem_three_uppers (ID 68):

  cab_1  (far-left)   — LOCKED at reset (friction-clamped hinges).
  cab_2  (middle)     — holds the banana (target for instruction 1).
  cab_3  (far-right)  — holds the ketchup (target for instruction 2).

Robot spawns facing cab_1. Both target objects are placed at reset;
because the chain's first instruction only inspects cab_2, perception
never sees inside cab_3 during instruction 1 — so the ketchup
remains genuinely "undiscovered" by the planner's memory until
instruction 2.

Between chain members the eval harness calls
``reset_for_chain_followup`` to (a) re-close all three cabinet doors,
(b) re-apply cab_1's friction lock, and (c) return the robot to its
cab_1 spawn so instruction 2 starts from the same physical state as
instruction 1. This makes the memory test fair: the planner is
facing cab_1 again and must remember from instruction 1 that cab_1
is locked, then route past it to cab_3 (the only unexplored cabinet).

Chain shape:
  1. find_banana   — planner tries cab_1 (locked → fails), recovers
                     to cab_2, finds banana. Memory now records
                     "cab_1 locked/stuck" and "cab_2 contents:
                     banana + fruit distractors". cab_3 stays
                     unexplored.
  2. find_ketchup  — reset_for_chain_followup just fired. Planner
                     faces cab_1; memory says cab_1 is locked and
                     cab_2 doesn't contain ketchup. Only unexplored
                     cabinet is cab_3 — planner routes there.
"""
import os

import numpy as np

import robocasa
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType
from robocasa.environments.kitchen.messymem.messymem_locked_left_cabinet import (
    MessymemLockedLeftCabinet,
)


class MessymemLockedChain(MessymemLockedLeftCabinet):
    """Three-cabinet locked-left variant — cab_1 locked, banana in cab_2,
    ketchup in cab_3. Provides a chain-followup hook so instruction 2
    starts from the same robot pose / door state as instruction 1.
    """

    # Cab_2 distractors so cab_2 isn't a one-item "free win" — the
    # planner has to actually inspect it to confirm the banana.
    _CAB2_OTHER_OBJECTS = ["apple", "lime", "lemon", "pear"]

    # ── Fixture references ────────────────────────────────────────────

    def _setup_kitchen_references(self):
        # Skip MessymemLockedLeftCabinet._setup_kitchen_references —
        # it only knows about cab_1 / cab_2. Go straight to the
        # Kitchen base class so we control the registration order.
        from robocasa.environments.kitchen.kitchen import Kitchen
        Kitchen._setup_kitchen_references(self)

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.cab3 = self.register_fixture_ref("cab3", dict(id="cab_3"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )
        # Robot spawns facing cab_1 — the locked one. The planner's
        # natural first attempt is OpenCabinet on cab_1; the lock
        # guarantees it fails, seeding "cab_1 stuck/locked" into
        # task_history before the chain's second instruction.
        self.init_robot_base_ref = self.cab1

    # ── Scene initialisation ──────────────────────────────────────────

    def _setup_scene(self):
        # Don't call MessymemLockedLeftCabinet._setup_scene — it only
        # closes cab_1 / cab_2 and would skip cab_3. Go to the base
        # Kitchen instead.
        from robocasa.environments.kitchen.kitchen import Kitchen
        Kitchen._setup_scene(self)
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)
        self.cab3.close_door(env=self)
        if self._LOCK_CAB1:
            self._lock_cabinet_doors(self.cab1)
        else:
            print("[LockedChain] _LOCK_CAB1=False — cab_1 left unlocked "
                  "(diffusion-policy isolation test).")

    # ── Object configs ────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        """Place banana + distractors in cab_2 (middle), ketchup in
        cab_3 (right).

        Both targets are CENTRED so they straddle the door seam and stay
        at least partly exposed whichever door the diffusion OpenCabinet
        policy manages to pull. They used to be pinned to the LEFT half,
        on the assumption that partial opening meant the left door — but
        which door opens is a coin flip, and the pin only paid off on one
        side of it. From pilot 20260806-092141:

            s42  cab_2  "opened the LEFT door"   -> banana and a lime    PASS
            s44  cab_2  "opened the LEFT door"   -> an apple and a banana PASS
            s45  cab_2  "opened the RIGHT door"  -> a lime and an apple   FAIL

        In s45 the left door stayed shut over the left-pinned banana,
        Inspect reported "lime, apple", and the planner concluded the
        banana was elsewhere — then continued into cab_3, opening the
        cabinet that holds instruction 2's target and contaminating the
        chain. One coin flip produced both the instruction-1 failure and
        the leak.

        Distractors stay constrained to the right half of cab_2 so the
        placement sampler doesn't collide with the banana: they sample at
        pos=(0.5) size=(0.45), roughly x in [0.28, 0.72], while a centred
        banana at size=(0.30) occupies about [-0.15, 0.15].

        +x is the right half, -x is the left half in this placer.
        """
        cfgs = []

        # cab_2 distractors — pinned to the right half so they don't
        # compete for the left-side spot reserved for the banana.
        for name in self._CAB2_OTHER_OBJECTS:
            cfgs.append(dict(
                name=name,
                obj_groups=name,
                graspable=True,
                placement=dict(
                    fixture=self.cab2,
                    size=(0.45, 0.50),
                    pos=(0.5, 0.0),   # right half, full depth
                ),
            ))

        # Banana in cab_2 (middle cabinet). Centred laterally so either
        # door reveals it; mid-depth so the wrist sweep catches it
        # without back-of-shelf occlusion.
        cfgs.append(dict(
            name="banana",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/banana/banana_8/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=(0.30, 0.25),
                pos=(0.0, 0.0),
            ),
        ))

        # Ketchup in cab_3 (right cabinet). Present from reset; the
        # chain's first instruction only inspects cab_2, so perception
        # never observes cab_3's contents until instruction 2's
        # planner navigates there.
        cfgs.append(dict(
            name="ketchup",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/ketchup/ketchup_7/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab3,
                size=(0.30, 0.25),
                pos=(0.0, 0.0),
            ),
        ))
        return cfgs

    # ── Chain-followup hook ───────────────────────────────────────────

    def reset_for_chain_followup(self, env=None):
        """Re-apply cab_1's lock and return the robot to its cab_1
        spawn. Called between chain members by the eval harness's
        setup hook so instruction 2 starts from the same robot pose
        as instruction 1. Door states are preserved — whatever cab_2
        / cab_3 looked like at the end of instruction 1 (typically
        cab_2 open, cab_3 closed) carries over to instruction 2, so
        the SG's recorded open/closed state matches physical reality.

        Idempotent across repeated invocations within the same trial.
        The chain's reset suppression keeps the SG / KF memory intact.

        *env* is accepted for harness symmetry with other env_method
        hooks but unused (this method operates on ``self``).
        """
        # Re-apply cab_1's friction lock as a defensive measure in
        # case anything during instruction 1 disturbed the friction /
        # damping / stiffness model used by _lock_cabinet_doors.
        if self._LOCK_CAB1:
            try:
                self._lock_cabinet_doors(self.cab1)
            except Exception as e:
                print(f"[LockedChain] re-lock cab_1 during followup "
                      f"failed: {e!r}")
        # Return the robot to its cab_1 spawn so the memory test is
        # fair (planner faces cab_1, must remember it's locked).
        try:
            EnvUtils.set_robot_to_position(self, self.init_robot_base_pos)
        except Exception as e:
            print(f"[LockedChain] robot teleport during followup "
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
        print("[LockedChain] chain followup: door states preserved, "
              "cab_1 re-locked, robot returned to cab_1 spawn.")
