"""
messymem_two_cabinets.py — MessyMem two-cabinet scene and task definition.

Scene: layout_messymem_001 (normal-size kitchen, layout ID 61).

Cabinet contents (both start CLOSED at reset):
  cab_1  (left upper hinge cabinet)  — pantry items:
      ketchup, cereal, spaghetti_box, canned_food, mayonnaise

  cab_2  (right upper hinge cabinet) — fruits:
      apple, avocado, lime, lemon, banana, pineapple

Counter distractors (on counter_main, near each cabinet):
  Below cab_1: orange, baguette
  Below cab_2: onion, bagged_food

Objectives grow in robocasa_sim.py (Objectives 2-4):
  - Obj 2: composite task with NavigateKitchen + OpenCabinet + pineapple detection
  - Obj 3: Inspect primitive
  - Obj 4: full MemoryServer / policy server integration
"""

from robocasa.environments.kitchen.kitchen import *


class MessymemTwoCabinets(Kitchen):
    """
    MessyMem Two-Cabinet scene task.

    Sets up the messymem_001 kitchen with specific objects in two upper
    cabinets and on the counter below them.  Both cabinets start CLOSED;
    objects inside are revealed as each cabinet door is opened.

    The high-level goal is: "Open the upper cabinets and find the pineapple."
    Composite success conditions are added in Objective 2.
    """

    def __init__(self, *args, **kwargs):
        # Include "aigen" registry so aigen-only categories (pineapple,
        # spaghetti_box) are resolvable.  The default is ("objaverse",
        # "lightwheel") which silently produces an empty valid-category list
        # for aigen-only objects and crashes on rng.choice([]).
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    def _setup_kitchen_references(self):
        """
        Register named references to cab_1, cab_2, and the back-wall counter.

        Fixture names in self.fixtures follow the pattern:
            {yaml_name}_{group_name}
        e.g.  "cab_1" under "main_group" → "cab_1_main_group"

        get_fixture(str) uses substring matching so "cab_1" reliably finds
        "cab_1_main_group" without hardcoding the group suffix.
        """
        super()._setup_kitchen_references()

        self.cab1 = self.register_fixture_ref("cab1", dict(id="cab_1"))
        self.cab2 = self.register_fixture_ref("cab2", dict(id="cab_2"))

        # Back-wall counter that both upper cabinets sit above.
        # register_fixture_ref caches the result so the ref=self.cab1 hint
        # is only used the first time to find the right counter.
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.cab1),
        )

        # Robot spawns facing cab_1 (leftmost upper cabinet).
        self.init_robot_base_ref = self.cab1

    # ── Scene initialisation ──────────────────────────────────────────────────

    def _setup_scene(self):
        """
        Place objects (super()), close both cabinets, then orient tall objects.
        """
        super()._setup_scene()
        self.cab1.close_door(env=self)
        self.cab2.close_door(env=self)

    # ── Language metadata ─────────────────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the upper cabinets and find the pineapple."
        return ep_meta

    # ── Object placements ─────────────────────────────────────────────────────

    def _get_obj_cfgs(self):
        """
        Define all graspable/interactable objects for this scene.

        Counter placement follows the pattern from restock_canned_food.py:
          - First object uses pos=("ref", -1.0) to anchor near the target
            cabinet; subsequent objects in the same area use reuse_region_from.
        Cabinet placement follows the pattern from stack_cans.py:
          - size=(1.0, 0.15) lets the placer scatter objects across the full
            cabinet width in a thin back-of-shelf strip.
          - All objects in the same cabinet share the same spec; the placer
            handles collision avoidance automatically.
        """
        cfgs = []

        # ── Counter below cab_1: orange, baguette ─────────────────────────────
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
                reuse_region_from="orange",   # same counter strip near cab_1
                size=(1.0, 0.30),
                pos=(0, 0),
            ),
        ))

        # ── Counter below cab_2: onion, bagged_food ───────────────────────────
        # bagged_food has graspable=False in the registry — it is a visual
        # distractor, not a pick-place target.
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
                reuse_region_from="onion",    # same counter strip near cab_2
                size=(1.0, 0.30),
                pos=(0, 0),
            ),
        ))

        # ── Inside cab_1: pantry items ────────────────────────────────────────
        # pos=(None, None): fully random placement inside the cabinet —
        # the placer samples valid non-overlapping positions automatically.
        # size=(1.0, 0.50): allow placement across the full width and half the
        # depth, giving maximum freedom with 5 objects.
        # graspable=False for spaghetti_box (non-graspable in registry).
        cab1_items = [
            ("ketchup",       "ketchup",       True),
            ("cereal",        "cereal",        True),
            ("spaghetti_box", "spaghetti_box", False),
            ("canned_food",   "canned_food",   True),
            ("mayonnaise",    "mayonnaise",    True),
        ]
        for name, group, graspable in cab1_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab1,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        # ── Inside cab_2: fruits ──────────────────────────────────────────────
        # Same fully-random pattern. Pineapple is the visual detection target
        # (non-graspable in the registry; it just needs to appear in the scene).
        cab2_items = [
            ("apple",     "apple",     True),
            ("avocado",   "avocado",   True),
            ("lime",      "lime",      True),
            ("lemon",     "lemon",     True),
            ("banana",    "banana",    True),
            ("pineapple", "pineapple", False),
        ]
        for name, group, graspable in cab2_items:
            cfgs.append(dict(
                name=name,
                obj_groups=group,
                graspable=graspable,
                placement=dict(
                    fixture=self.cab2,
                    size=(1.0, 0.50),
                    pos=(None, None),
                ),
            ))

        return cfgs

    # ── Success condition (stub) ──────────────────────────────────────────────

    def _check_success(self):
        """
        Placeholder — always returns False.

        Objective 2 will add composite success conditions:
          - NavigateKitchen sub-goal
          - OpenCabinet sub-goal
          - High-level: pineapple detected and added to scene graph
        The env runs with ignore_done=True so this never terminates early.
        """
        return False
