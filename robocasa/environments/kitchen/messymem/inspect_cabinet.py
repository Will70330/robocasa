"""
inspect_cabinet.py — MessyMem Inspect primitive (stub).

Objective 3 will replace _check_success with real gripper-facing logic.
For now the task auto-succeeds so the mock plan can advance through it.
"""

from robocasa.environments.kitchen.kitchen import *


class InspectCabinet(Kitchen):
    """
    Inspect Cabinet: placeholder atomic task for the Inspect primitive.

    The robot is placed in front of the target cabinet.  Success is always
    True (stub) — Objective 3 will implement the actual gripper-orient logic
    and a hold-position timer.
    """

    def __init__(self, fixture_id=FixtureType.CABINET_WITH_DOOR, *args, **kwargs):
        self.fixture_id = fixture_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fxtr = self.register_fixture_ref("fxtr", dict(id=self.fixture_id))
        self.init_robot_base_ref = self.fxtr

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Inspect the {self.fxtr.nat_lang}."
        return ep_meta

    def _check_success(self):
        # Auto-succeed — Objective 3 will add real success logic.
        return True
