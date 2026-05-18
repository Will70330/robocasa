"""
messymem_locked_left_cabinet.py — MessyMem find-the-banana task with the
LEFT upper cabinet (cab_1) locked.

Scenario: banana is in cab_2 (right cabinet). cab_1 (left cabinet) is
physically locked at reset — its hinge joints have their MuJoCo range
clamped to a tiny window so the OpenCabinet policy can apply force but
the doors can't move. Robot spawns facing cab_1 (the locked one), so the
planner's natural first attempt is to open cab_1, fail, and then
re-plan to cab_2. Tests the planner's recovery loop.

Designed to run on layout_messymem_uppers_only (ID 66) — same geometry
as MessymemTwoCabinets, just a different task class with locked cab_1.
"""
import os

import robocasa
from robocasa.environments.kitchen.kitchen import *


class MessymemLockedLeftCabinet(Kitchen):
    """
    Find-the-banana task with cab_1 (left) locked.

    Banana is fixed in cab_2; the planner can only succeed by opening
    cab_2 — but it doesn't know cab_1 is locked until it tries.
    """

    def __init__(self, *args, **kwargs):
        # aigen registry for spaghetti_box etc., same as MessymemTwoCabinets.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    # cab_1 is locked — leave it empty. Nothing in it can ever be observed
    # so populating it adds object-sampling failure modes (each
    # obj_groups entry needs to resolve to non-empty valid_categories
    # after the registry+graspable filter) for zero benefit.
    # Distractors in cab_2 alongside the banana.
    _CAB2_OTHER_OBJECTS = ["apple", "lime", "lemon", "pear"]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab1)
        )
        # Robot spawns facing cab_1 — the locked cabinet. Planner's
        # natural first move is to try cab_1, discover the failure, then
        # re-plan to cab_2 (where the banana actually is).
        self.init_robot_base_ref = self.cab1

    # ── Spawn offset (matches MessymemTwoCabinets) ────────────────────────────
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
        # Both cabinets start closed.
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)
        self._lock_cabinet_doors(self.cab1)

    def _lock_cabinet_doors(self, cab):
        """Make the cabinet's hinge joints behave like a real lock —
        very high Coulomb friction (jnt_frictionloss). The door resists
        any motion below the friction threshold, but applies no
        restoring SPRING force. Critical difference from a stiff spring:

          - Spring locks fight gripper contact reactively → instant
            pushback when the gripper touches the handle → contact
            instability → gripper bounces off / slips. Policy never
            establishes a stable grasp.
          - Friction locks only fight motion that's already happening.
            Static contact (gripper closing around the handle) doesn't
            trigger friction; friction only resists when the policy
            pulls. Door doesn't move, gripper grasps normally.

        The OpenCabinet policy's reach + gripper-close motions run as
        if the door were a normal closed cabinet. Only when the policy
        starts PULLING does the friction lock kick in and prevent door
        motion — exactly the "tried to open, mechanism didn't yield"
        sequence the InteractionAnalyzer needs.
        """
        sim = self.sim
        for short in ("leftdoorhinge", "rightdoorhinge"):
            full_name = f"{cab.name}_{short}"
            try:
                joint_id = sim.model.joint_name2id(full_name)
            except (ValueError, KeyError):
                print(f"[LockedLeft] joint {full_name!r} not found — "
                      f"can't lock; cabinet will open normally.")
                continue
            # Reset to closed pose at the start of every trial.
            qpos_addr = sim.model.get_joint_qpos_addr(full_name)
            sim.data.qpos[qpos_addr] = 0.0
            # "Stuck door" model: SOFT spring + HEAVY damping + LIGHT
            # Coulomb friction.
            #
            #   - stiffness = 300 Nm/rad → at qpos=0 the spring force
            #     is exactly 0, so gripper contact (which barely
            #     deflects the joint) doesn't trigger pushback. Spring
            #     grows linearly with deflection so a sustained pull
            #     reaches equilibrium at qpos = T/300; e.g. 30 Nm pull
            #     → qpos ≈ 0.1 rad ≈ 6°, well below the 0.7-rad
            #     "open" threshold. Door visibly yields a bit (so the
            #     InteractionAnalyzer sees the attempt), but never
            #     enough to count as opened.
            #
            #   - damping = 100 → critically-damped at this stiffness,
            #     so the door doesn't oscillate when the policy
            #     releases. Settles back to qpos≈0 in <0.3 s.
            #
            #   - frictionloss = 50 → adds static stiction so the door
            #     doesn't drift under tiny gripper-contact forces.
            #     Far below stiffness contribution, so gripper grasp
            #     stays unaffected.
            #
            # NO qpos clamp in _post_action — the clamp caused
            # intra-step physics to snap back at frame boundaries,
            # moving the handle under the gripper and breaking the
            # grasp. Spring + damping reach equilibrium smoothly,
            # gripper stays engaged.
            dof_addr = sim.model.jnt_dofadr[joint_id]
            sim.model.jnt_stiffness[joint_id] = 300.0
            sim.model.dof_damping[dof_addr] = 100.0
            sim.model.dof_frictionloss[dof_addr] = 50.0
            sim.model.qpos_spring[joint_id] = 0.0
            print(f"[LockedLeft] locked {full_name!r} via stuck-door "
                  f"model: stiffness=300, damping=100, "
                  f"frictionloss=50, qpos_spring=0.")

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Find the banana."
        ep_meta["target_obj"] = "banana"
        # Surface the locked-cabinet info so logs / dashboards can show it.
        # The planner doesn't see this field — it has to discover the lock
        # by trying and failing.
        ep_meta["locked_fixtures"] = ["cab_1"]
        ep_meta["target_fixture"] = "cab_2"
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        cfgs = []

        # cab_2 distractors.
        for name in self._CAB2_OTHER_OBJECTS:
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
        # The goal object: banana, pinned to a tight front-centre region of
        # cab_2's shelf so the wrist-cam sweep can reliably pick it up
        # (mirrors MessymemTwoCabinets' banana placement).
        cfgs.append(dict(
            name="banana",
            obj_groups=os.path.join(
                robocasa.models.assets_root,
                "objects/objaverse/banana/banana_8/model.xml",
            ),
            graspable=True,
            placement=dict(
                fixture=self.cab2,
                size=(0.5, 0.20),
                pos=(0.4, 1.0),
            ),
        ))

        return cfgs

    # ── Success condition ─────────────────────────────────────────────────────

    # Threshold for "the cabinet is open enough to see/grasp the banana".
    # Matches MessymemTwoCabinets._OPEN_THRESHOLD.
    _OPEN_THRESHOLD = 0.70

    def _check_success(self):
        """Banana is in cab_2; success = cab_2 is open past the threshold.
        cab_1 is locked so it physically cannot satisfy this even if the
        planner tries hard."""
        return self.cab2.is_open(env=self, th=self._OPEN_THRESHOLD)
