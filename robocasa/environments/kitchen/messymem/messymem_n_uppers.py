"""
messymem_n_uppers.py — upper-cabinet-row scenes for long-horizon memory
evaluations, in 10-, 15- and 20-cabinet sizes.

  MessymemTenCabinets      layout 72  layout_messymem_ten_uppers
  MessymemFifteenCabinets  layout 73  layout_messymem_fifteen_uppers
  MessymemTwentyCabinets   layout 74  layout_messymem_twenty_uppers

All three share one implementation and differ only in ``_N_CABINETS``.
Each layout is a row of upper hinge cabinets 1.40 m apart starting at
x = 1.75, one continuous counter beneath them, nothing else in the room.

Cabinet ids are ZERO-PADDED TO THREE DIGITS — ``cab_001`` … ``cab_020``.
This is not cosmetic. ``Kitchen.get_fixture`` resolves a name by
SUBSTRING match (``id in name``), so a variable-width id matches any
longer id it is a prefix of: ``cab_1`` also matches
``cab_10_main_group``, and two-digit ``cab_10`` would match
``cab_100_main_group``. The lookup silently returns whichever it picks,
which puts targets in the wrong cabinet and spawns the robot at the
wrong end of the row. Fixed-width ids cannot prefix one another.

Contents follow a HYBRID scheme — fixed targets, random everything else:

  * cab_001 … cab_004 always hold a fixed pair of find-targets. These
    pairs never move and never change between seeds or between the three
    sizes, so a find-chain's ground truth is stable:

        cab_001 : lemon,   mug
        cab_002 : apple,   pear
        cab_003 : mustard, orange
        cab_004 : cereal,  pineapple

  * Each target cabinet is topped up with 1-3 random fillers, giving the
    3-5 items per cabinet the scene calls for.

  * Every cabinet after cab_004 is pure scenery, drawn fresh from
    ``self.rng`` on every reset. Between a third and two thirds of them
    are left COMPLETELY EMPTY; the rest get 3-5 random fillers. The
    count scales with the row length, so the 10-, 15- and 20-cabinet
    scenes stay comparably sparse.

  * Counter space below each cabinet gets 0-3 random fillers, decided
    independently of whether the cabinet above it is empty. At least one
    empty cabinet is guaranteed to have a bare counter beneath it, and at
    least one empty cabinet is guaranteed to have items beneath it — so
    "cabinet is empty" and "counter below is empty" are never
    correlated cues the planner could exploit.

Fillers are drawn from a pool that contains NO target item, so a random
draw can never create a second copy of a find-target somewhere else in
the room. Every group in the pool is already proven to load in existing
messymem tasks — no new assets, no render-check pass.

Randomisation is seeded: ``self.rng`` is ``np.random.default_rng(seed)``
(robosuite ``base.py:145``), so a given seed reproduces a given kitchen
exactly. The realised assignment is published through ``get_ep_meta()``
under ``cabinet_contents`` / ``counter_contents`` so an eval config can
recover per-episode ground truth without re-deriving it.

These classes define the SCENE only. They carry no goal of their own —
instruction text, success checks and chain structure live in the eval
config, and ``_check_success`` is deliberately inert (see below).
"""
import os

import numpy as np

import robocasa
from robocasa.environments.kitchen.kitchen import *


# Separator between a category and a variant tag in _FIXED_CONTENTS /
# _FIXED_COUNTER / _PINNED_MODELS. `_PINNED_MODELS` maps one model to one
# key, which is fine while a scene wants a single mug — but an
# appearance-matching task needs THREE mugs that are unmistakably different
# and all still called "mug". Tagging lets each placement name its own
# model (`mug#red`, `mug#teal`, `mug#yellow`) while everything that reads
# the scene — contents lists, object names, the graspable filter — keeps
# seeing the bare category.
_VARIANT_SEP = "#"


def _base_group(spec):
    """`mug#red` -> `mug`; an untagged category is returned unchanged."""
    return spec.split(_VARIANT_SEP, 1)[0]


class MessymemUpperRow(Kitchen):
    """Base class for the upper-cabinet-row scenes. Subclasses set
    ``_N_CABINETS`` and are paired with the matching layout id.
    """

    # Overridden per size. Must match the number of cabinets in the
    # layout yaml the env is run against.
    _N_CABINETS = 10

    # Width of the zero-padded numeric suffix in cabinet ids. Three
    # digits stays collision-free up to 999 cabinets; see the module
    # docstring for why fixed width is required at all.
    _ID_WIDTH = 3

    # ── Fixed targets ─────────────────────────────────────────────────────────
    # Pinned to cab_001..cab_004 so find-chain ground truth is stable
    # across seeds AND across the three row lengths — the same chain can
    # be run on 10, 15 or 20 cabinets and only the search space changes.
    # Concentrating them at the near end leaves the rest of the row as
    # "negative space" a memory-enabled planner can exclude, which is
    # what makes the exclusion/absence phases of a long chain measurable.
    _TARGETS = {
        "cab_001": ["lemon", "mug"],
        "cab_002": ["apple", "pear"],
        "cab_003": ["mustard", "orange"],
        "cab_004": ["cereal", "pineapple"],
    }

    # ── Filler pool ───────────────────────────────────────────────────────────
    # Disjoint from _TARGETS by construction — a filler draw can never
    # duplicate a find-target elsewhere in the room, which would make the
    # target ambiguous. Sampled WITHOUT replacement within a cabinet, so
    # no cabinet holds two of the same thing; repeats across cabinets are
    # fine and in fact desirable (they stop item identity alone from
    # localising a cabinet).
    _FILLER_POOL = ["ketchup", "bagged_food", "banana", "mayonnaise", "lime", "onion"]

    # Groups whose meshes are too large / awkward for the grasp checker.
    # Matches the graspable flags used in MessymemManyCabinets.
    _NON_GRASPABLE = {"bagged_food", "pineapple"}

    # ── Randomisation bounds ──────────────────────────────────────────────────
    _N_FILLERS_IN_TARGET_CAB = (1, 4)   # rng.integers → 1-3, total 3-5 with targets
    _N_ITEMS_IN_FILLER_CAB   = (3, 6)   # rng.integers → 3-5
    _N_ON_COUNTER            = (0, 4)   # rng.integers → 0-3
    # Fraction of the scenery cabinets left empty. Expressed as a
    # fraction rather than a count so the three row lengths stay
    # comparably sparse: 6 scenery cabinets → 2-4 empty (the 10-cabinet
    # scene), 11 → 4-7, 16 → 5-11.
    _EMPTY_FRACTION = (1 / 3, 2 / 3)

    def __init__(self, *args, **kwargs):
        # mayonnaise / mustard are lightwheel-only and pineapple is
        # aigen-only, so all three registries are required for the pool
        # and target lists to resolve. Matches MessymemManyCabinets.
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        # Realised per-episode assignment, filled in by _get_obj_cfgs and
        # read back by get_ep_meta. Initialised here so get_ep_meta is
        # safe to call before the first reset.
        self._cabinet_contents = {}
        self._counter_contents = {}
        super().__init__(*args, **kwargs)

    # ── Fixture references ────────────────────────────────────────────────────

    @property
    def cab_ids(self):
        return [
            f"cab_{i:0{self._ID_WIDTH}d}"
            for i in range(1, self._N_CABINETS + 1)
        ]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # Register every cabinet under BOTH the ref name used by
        # register_fixture_ref and a bare attribute matching the layout
        # id, because the policy server's _resolve_fixture tries
        # getattr(base, "cab_004") first when the planner emits
        # "OpenCabinet:cab_004".
        self.cabs = []
        for cab_id in self.cab_ids:
            cab = self.register_fixture_ref(cab_id, dict(id=cab_id))
            setattr(self, cab_id, cab)
            self.cabs.append(cab)
        # cab1 … cabN aliases, matching the naming in the other messymem
        # tasks (MessymemLockedChain uses self.cab1 / self.cab2 / self.cab3).
        for i, cab in enumerate(self.cabs, start=1):
            setattr(self, f"cab{i}", cab)

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabs[0])
        )

        # Robot spawns facing cab_001, the near end of the row — the same
        # end the chain's sub-tasks start from, so every sub-task begins
        # from an identical pose.
        self.init_robot_base_ref = self.cabs[0]

    # ── Spawn offset ──────────────────────────────────────────────────────────
    # Pull the base 0.15 m back off the counter so it clears the cabinets
    # when driving east-west. Same value as MessymemTwoCabinets /
    # MessymemManyCabinets / MessymemLockedLeftCabinet.
    _SPAWN_PULLBACK = 0.15

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        yaw = self.init_robot_base_ori_anchor[2]
        self.init_robot_base_pos_anchor[:2] -= self._SPAWN_PULLBACK * np.array(
            [np.cos(yaw), np.sin(yaw)]
        )

    # ── Scene initialisation ──────────────────────────────────────────────────

    def _setup_scene(self):
        super()._setup_scene()
        for cab in self.cabs:
            cab.close_door(env=self)
        self._apply_locks()

    # ── Locking ───────────────────────────────────────────────────────────────
    # Cabinet ids that start the episode locked. Empty by default, so the
    # randomised scenes are entirely unlocked and nothing pre-existing
    # changes behaviour. Subclasses override.
    _LOCKED_CABS = ()

    def _apply_locks(self):
        """Lock every cabinet in ``_LOCKED_CABS``. Idempotent."""
        self._unlocked_at_runtime = set(getattr(self, "_unlocked_at_runtime", set()))
        for cab_id in self._LOCKED_CABS:
            if cab_id in self._unlocked_at_runtime:
                continue
            cab = getattr(self, cab_id, None)
            if cab is None:
                print(f"[{type(self).__name__}] lock target {cab_id!r} not found.")
                continue
            self._lock_cabinet_doors(cab)

    def _lock_cabinet_doors(self, cab):
        """Make the cabinet's hinges behave like a stuck door — soft spring,
        heavy damping, light Coulomb friction.

        Lifted verbatim (parameters included) from
        MessymemLockedLeftCabinet._lock_cabinet_doors, which tuned these
        values so the OpenCabinet policy can still grasp the handle and
        pull — the door simply never reaches the 0.5 rad "open" threshold.
        A stiff spring instead fights gripper contact reactively and
        destabilises the grasp, which reads as a grasp failure rather than
        the "tried to open, mechanism didn't yield" signal the
        InteractionAnalyzer needs.
        """
        sim = self.sim
        for short in ("leftdoorhinge", "rightdoorhinge"):
            full_name = f"{cab.name}_{short}"
            try:
                joint_id = sim.model.joint_name2id(full_name)
            except (ValueError, KeyError):
                print(f"[{type(self).__name__}] joint {full_name!r} not found — "
                      f"can't lock; cabinet will open normally.")
                continue
            qpos_addr = sim.model.get_joint_qpos_addr(full_name)
            sim.data.qpos[qpos_addr] = 0.0
            dof_addr = sim.model.jnt_dofadr[joint_id]
            sim.model.jnt_stiffness[joint_id] = 100.0
            sim.model.dof_damping[dof_addr] = 40.0
            sim.model.dof_frictionloss[dof_addr] = 5.0
            sim.model.qpos_spring[joint_id] = 0.0

    def unlock_cabinet(self, cab_id=None, env=None):
        """Release a locked cabinet mid-run so a later sub-task can reach
        its contents. Exposed as an ``env_method`` setup hook.

        OFF BY DEFAULT — nothing calls this unless an eval config names it
        in an instruction's ``setup.env_method``, so no existing task
        changes behaviour. Restores the hinge parameters to MuJoCo's
        defaults for a free-swinging door (no spring, light damping, no
        friction), which is how an unlocked cab_top hinge behaves.

        The unlock is recorded in ``_unlocked_at_runtime`` so
        ``_apply_locks`` (called from _setup_scene and from
        reset_for_chain_followup) does not silently re-lock it for the
        remainder of the chain.

        *env* is accepted for symmetry with other env_method hooks.
        """
        cab_id = cab_id or getattr(self, "_UNLOCK_TARGET", None)
        tag = type(self).__name__
        if cab_id is None:
            print(f"[{tag}] unlock_cabinet called with no target; ignoring.")
            return
        cab = getattr(self, cab_id, None)
        if cab is None:
            print(f"[{tag}] unlock target {cab_id!r} not found; ignoring.")
            return
        sim = self.sim
        for short in ("leftdoorhinge", "rightdoorhinge"):
            full_name = f"{cab.name}_{short}"
            try:
                joint_id = sim.model.joint_name2id(full_name)
            except (ValueError, KeyError):
                continue
            dof_addr = sim.model.jnt_dofadr[joint_id]
            sim.model.jnt_stiffness[joint_id] = 0.0
            sim.model.dof_damping[dof_addr] = 1.0
            sim.model.dof_frictionloss[dof_addr] = 0.0
            sim.model.qpos_spring[joint_id] = 0.0
        self._unlocked_at_runtime = set(
            getattr(self, "_unlocked_at_runtime", set())) | {cab_id}
        self.sim.forward()
        print(f"[{tag}] unlocked {cab_id} mid-run — door now swings freely.")

    # ── Object placements ─────────────────────────────────────────────────────

    def _draw_fillers(self, n):
        """Draw *n* distinct groups from the filler pool."""
        n = int(min(n, len(self._FILLER_POOL)))
        return list(self.rng.choice(self._FILLER_POOL, size=n, replace=False))

    def _empty_count_bounds(self, n_scenery):
        """Half-open [low, high) bounds for how many scenery cabinets are
        left empty, scaled to the row length. Floored at 2 so the
        bare-counter / stocked-counter guarantee below always has two
        distinct cabinets to work with.
        """
        lo_f, hi_f = self._EMPTY_FRACTION
        low = max(2, round(n_scenery * lo_f))
        high = max(low + 1, round(n_scenery * hi_f) + 1)
        return low, min(high, n_scenery + 1)

    def _plan_contents(self):
        """Decide, for this episode, what goes in each cabinet and on the
        counter below it. Returns (cabinet_contents, counter_contents),
        both dicts keyed by cabinet id.

        Called from _get_obj_cfgs, which robocasa re-invokes on every
        placement retry (up to 50, see kitchen.py:650). Re-drawing on
        each attempt is intentional: a cabinet whose sampled items can't
        be packed gets a fresh draw rather than deadlocking the reset.
        """
        scenery_cabs = [c for c in self.cab_ids if c not in self._TARGETS]

        cabinet_contents = {}

        # Target cabinets: fixed pair + 1-3 random fillers → 3-5 items.
        for cab_id, targets in self._TARGETS.items():
            n_extra = int(self.rng.integers(*self._N_FILLERS_IN_TARGET_CAB))
            cabinet_contents[cab_id] = list(targets) + self._draw_fillers(n_extra)

        # Scenery cabinets: a scaled random share are left completely
        # empty, the rest get 3-5 fillers.
        n_empty = int(self.rng.integers(*self._empty_count_bounds(len(scenery_cabs))))
        empty_cabs = list(
            self.rng.choice(scenery_cabs, size=n_empty, replace=False)
        )
        for cab_id in scenery_cabs:
            if cab_id in empty_cabs:
                cabinet_contents[cab_id] = []
            else:
                n_items = int(self.rng.integers(*self._N_ITEMS_IN_FILLER_CAB))
                cabinet_contents[cab_id] = self._draw_fillers(n_items)

        # Counter below each cabinet: 0-3 fillers, drawn independently of
        # what's above.
        counter_contents = {
            cab_id: self._draw_fillers(int(self.rng.integers(*self._N_ON_COUNTER)))
            for cab_id in self.cab_ids
        }

        # Guarantee both flavours of "skipped" cabinet exist, so the scene
        # always contains at least one of each case the eval cares about:
        # an empty cabinet over a bare counter, and an empty cabinet over
        # a stocked one. Without this the random draw can produce a seed
        # where empty cabinets happen to be uniformly bare (or uniformly
        # stocked), which would let a planner treat counter clutter as a
        # proxy for cabinet contents. n_empty >= 2 always, so there are
        # always two distinct cabinets to assign.
        bare, stocked = self.rng.permutation(empty_cabs)[:2]
        counter_contents[bare] = []
        if not counter_contents[stocked]:
            counter_contents[stocked] = self._draw_fillers(
                int(self.rng.integers(1, self._N_ON_COUNTER[1]))
            )

        return cabinet_contents, counter_contents

    # ── Fixed-scene override ──────────────────────────────────────────────────
    # When _FIXED_CONTENTS is non-empty the RNG is bypassed entirely and the
    # scene is byte-identical on every reset and every seed. Format:
    #
    #   _FIXED_CONTENTS = {cab_id: [(spec, (x, y), (w, d)[, shelf[, rot]]), ...]}
    #   _FIXED_COUNTER  = {cab_id: [spec, ...]}
    #
    # `spec` is a category name, optionally tagged with a variant —
    # "mug" or "mug#red". The tag selects a _PINNED_MODELS entry; the bare
    # category is what appears in contents lists and object names.
    #
    # (x, y) are the normalised in-cabinet placement coords the robocasa
    # sampler takes (env_utils.py) — -1 is the far left / back of the
    # sampling window, +1 the far right / front. They are how the
    # clutter-spatial signal is expressed: an item at x=0 flanked by items
    # at x=±0.35 is WEDGED, an item alone at x=-0.85 is ACCESSIBLE.
    # (w, d) is that item's sampling-window size in metres; smaller windows
    # pin the item more tightly to its (x, y).
    #
    # `shelf` names a reset region ("level1"); `rot` pins the yaw, which
    # only matters for items whose identity is printed on one face — a
    # cereal box is its brand from the front and a blank nutrition panel
    # from behind, and the default draw is ±45°.
    _FIXED_CONTENTS = {}
    _FIXED_COUNTER = {}

    # Category -> explicit model.xml, relative to robocasa.models.assets_root.
    #
    # WHY THIS EXISTS. Pinning a category in _FIXED_CONTENTS pins the item's
    # IDENTITY but not its MESH: sample_object(..., rng=self.rng)
    # (kitchen.py) still draws which variant of the category you get, and
    # variants differ enormously in size — robocasa's plate variants range
    # past 0.49 m across, cutting_board past 0.57 m deep. Two consequences:
    #
    #   - The scene is only reproducible at the category level. A different
    #     seed gives the same items in the same places, but different
    #     models, so appearance and footprint shift run to run.
    #   - Placement feasibility becomes RNG-dependent, and NON-LOCALLY so.
    #     Adding an object anywhere in the scene shifts every later draw,
    #     which is why a cabinet that packs perfectly on its own can break
    #     when an unrelated cabinet is populated first. Debugging that by
    #     staring at the failing cabinet gets you nowhere.
    #
    # Pin anything whose size or appearance a task depends on. Entries here
    # override the category draw entirely.
    _PINNED_MODELS = {}

    def _obj_spec(self, spec):
        """Model path if the entry is pinned, else the category name.

        Looks up the full spec first so a variant tag wins, then falls back
        to the bare category. See _base_group for the tag syntax.
        """
        rel = self._PINNED_MODELS.get(spec)
        if rel is None:
            rel = self._PINNED_MODELS.get(_base_group(spec))
        if rel is None:
            return _base_group(spec)
        return os.path.join(robocasa.models.assets_root, rel)

    # Groups that must remain pickable — `graspable=True` filters the
    # category to gripper-friendly variants. Restricted to items a pick
    # task actually targets, because the filter can empty out a category
    # and fail scene construction.
    _GRASPABLE = set()

    def _fixed_plan(self):
        # entry is (group, pos, size[, shelf[, rotation]]) — index rather
        # than unpack so every arity works. Variant tags are stripped: the
        # contents list is a list of CATEGORIES, which is what the eval
        # harness's ground_truth_contents is written against.
        contents = {c: [_base_group(e[0])
                        for e in self._FIXED_CONTENTS.get(c, [])]
                    for c in self.cab_ids}
        counter = {c: [_base_group(g) for g in self._FIXED_COUNTER.get(c, [])]
                   for c in self.cab_ids}
        return contents, counter

    def _get_obj_cfgs(self):
        if self._FIXED_CONTENTS:
            return self._get_obj_cfgs_fixed()

        cabinet_contents, counter_contents = self._plan_contents()
        self._cabinet_contents = cabinet_contents
        self._counter_contents = counter_contents

        cfgs = []
        for cab_id, cab in zip(self.cab_ids, self.cabs):
            targets = self._TARGETS.get(cab_id, [])

            for i, group in enumerate(cabinet_contents[cab_id]):
                if group in targets:
                    # Targets are pinned to the LEFT half of the cabinet.
                    # The diffusion OpenCabinet policy frequently opens
                    # only one of the two doors, and a left-pinned target
                    # stays visible in that case — the same convention
                    # MessymemLockedChain uses for its banana / ketchup.
                    placement_pos, placement_size = (-0.5, 0.0), (0.35, 0.35)
                else:
                    # Fillers take the right half, so the placement
                    # sampler isn't repeatedly colliding with the
                    # reserved target spot.
                    placement_pos, placement_size = (0.5, 0.0), (0.45, 0.50)

                cfgs.append(dict(
                    name=f"{cab_id}_in{i}_{group}",
                    obj_groups=group,
                    graspable=group not in self._NON_GRASPABLE,
                    placement=dict(
                        fixture=cab,
                        size=placement_size,
                        pos=placement_pos,
                    ),
                ))

            # Counter below this cabinet. Every object refs the cabinet
            # directly rather than reusing the first one's region:
            # reuse_region_from copies the whole counter-top region
            # (env_utils.py:1079), and with pos=(0, 0) on a counter this
            # long that centres the object at the counter's midpoint
            # instead of under this cabinet. Ref'ing each object to the
            # cabinet resolves inner_xpos through the "ref" branch
            # (env_utils.py:1168) and lands them all in the same window.
            for i, group in enumerate(counter_contents[cab_id]):
                cfgs.append(dict(
                    name=f"{cab_id}_on{i}_{group}",
                    obj_groups=group,
                    graspable=group not in self._NON_GRASPABLE,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(ref=cab),
                        size=(0.55, 0.25),
                        pos=("ref", -1.0),
                    ),
                ))

        return cfgs

    def _get_obj_cfgs_fixed(self):
        """Deterministic scene build from _FIXED_CONTENTS / _FIXED_COUNTER.

        Object names are ``<cab>_in<i>_<group>`` / ``<cab>_on<i>_<group>``,
        the same scheme the randomised path uses, so anything that parses
        names (the counter-placement check, debugging) works for both.
        A group repeated across cabinets is fine — the index keeps names
        unique — and repeats are load-bearing here: the same item in two
        cabinets with different neighbours is what a contextual
        "which one?" instruction disambiguates.

        PLACEMENT FAILURES ARE PERMANENT ON THIS PATH. robocasa retries
        _load_model up to 50 times (kitchen.py:650), but each retry calls
        this method again and gets a byte-identical answer, so a scene
        that cannot be packed once can never be packed — the run dies with
        "Ran _load_model() 50 times" and no indication of which object was
        at fault. The randomised path escapes by redrawing; this one
        cannot.

        The usual cause is a sampling window narrower than the object.
        `inner_size = min(outer_size, target_size)` (env_utils.py), so the
        window size given here is a HARD CEILING on the object's footprint
        — a plate or tray will not fit the 0.15-0.20 m windows that suit
        bottles and fruit, even alone in an empty cabinet. Either widen
        the window or add the group to _GRASPABLE, which filters the
        category to smaller gripper-friendly variants. Widening costs
        positional precision: the normalised x in _FIXED_CONTENTS spans
        `(outer - inner) / 2`, so a wider window means less spread
        between neighbours and a weaker wedged/accessible contrast.
        """
        cabinet_contents, counter_contents = self._fixed_plan()
        self._cabinet_contents = cabinet_contents
        self._counter_contents = counter_contents

        cfgs = []
        for cab_id, cab in zip(self.cab_ids, self.cabs):
            for i, entry in enumerate(self._FIXED_CONTENTS.get(cab_id, [])):
                # 3-tuple = bottom shelf (the historical form); optional 4th
                # element names a shelf region, optional 5th pins the yaw.
                spec, pos, size = entry[:3]
                group = _base_group(spec)
                shelf = entry[3] if len(entry) > 3 else None
                rotation = entry[4] if len(entry) > 4 else None
                placement = dict(fixture=cab, size=size, pos=pos)
                if rotation is not None:
                    # Default is a +-45 deg draw (env_utils.py), which is
                    # fine for anything that looks the same from every side
                    # and useless for anything whose identity is printed on
                    # one face. A scalar here fixes the yaw exactly; a
                    # (lo, hi) pair narrows the draw without fixing it.
                    placement["rotation"] = rotation
                if shelf is not None:
                    # Targeting a shelf ABOVE the bottom one needs both
                    # arguments. reset_region_names picks the shelf; z_range
                    # MUST be disabled because its default (0.45, 1.50)
                    # silently drops every region above 1.50 m — and in this
                    # layout the cabinets sit at z = 1.85, so level1 (1.72)
                    # survives but level2 (2.01) does not. Leave z_range at
                    # its default and a level2 request returns nothing, which
                    # surfaces as an unexplained placement failure rather
                    # than an error naming the shelf.
                    placement["sample_region_kwargs"] = dict(
                        reset_region_names=(shelf,), z_range=None)
                cfgs.append(dict(
                    name=f"{cab_id}_in{i}_{group}",
                    obj_groups=self._obj_spec(spec),
                    graspable=group in self._GRASPABLE,
                    placement=placement,
                ))
            for i, spec in enumerate(self._FIXED_COUNTER.get(cab_id, [])):
                group = _base_group(spec)
                cfgs.append(dict(
                    name=f"{cab_id}_on{i}_{group}",
                    obj_groups=self._obj_spec(spec),
                    graspable=group in self._GRASPABLE,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(ref=cab),
                        size=self._COUNTER_WINDOW,
                        pos=("ref", -1.0),
                    ),
                ))
        return cfgs

    # Per-cabinet counter footprint. Widened from the original (0.55, 0.25)
    # to fit plate-scale items, which are ~0.49 m across and could not fit a
    # 0.25 m depth at all. Cabinets are 1.4 m apart so neighbouring windows
    # still cannot overlap, and 0.40 stays inside the counter's 0.61 m
    # usable depth.
    _COUNTER_WINDOW = (0.70, 0.40)

    # ── Language / episode metadata ───────────────────────────────────────────

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Search the upper cabinets."
        # Realised ground truth for this episode. `targets` is constant;
        # the other two vary with the seed, so an eval config that needs
        # ground_truth_contents should read them from here rather than
        # hardcoding them.
        ep_meta["n_cabinets"] = self._N_CABINETS
        ep_meta["targets"] = {k: list(v) for k, v in self._TARGETS.items()}
        ep_meta["cabinet_contents"] = {
            k: list(v) for k, v in self._cabinet_contents.items()
        }
        ep_meta["counter_contents"] = {
            k: list(v) for k, v in self._counter_contents.items()
        }
        ep_meta["empty_cabinets"] = [
            k for k, v in self._cabinet_contents.items() if not v
        ]
        # Locked cabinets are surfaced for logs / dashboards only. The
        # planner never sees this — it has to discover a lock by trying
        # to open the door and failing.
        ep_meta["locked_fixtures"] = list(self._LOCKED_CABS)
        ep_meta["unlocked_at_runtime"] = sorted(
            getattr(self, "_unlocked_at_runtime", set()))
        return ep_meta

    # ── Chain-followup hook ───────────────────────────────────────────────────

    def reset_for_chain_followup(self, env=None):
        """Re-close every cabinet and return the robot to its cab_001 spawn.

        Invoked between chain members via an instruction's
        ``setup.env_method``, the same way MessymemLockedChain's hook is.
        Unlike that one there is no lock to re-apply — every cabinet here
        is a normal openable cabinet — so this only restores the physical
        starting state. Scene-graph / keyframe memory is left untouched;
        the harness's reset suppression is what carries it across
        sub-tasks.

        *env* is accepted for symmetry with other env_method hooks but
        unused (this operates on ``self``).
        """
        tag = type(self).__name__
        for cab in self.cabs:
            try:
                cab.close_door(env=self)
            except Exception as e:
                print(f"[{tag}] closing {cab.name} during followup failed: {e!r}")
        # Re-assert the locks defensively, in case anything during the
        # previous sub-task disturbed the hinge parameters. Cabinets
        # released via unlock_cabinet stay released — _apply_locks skips
        # anything in _unlocked_at_runtime.
        try:
            self._apply_locks()
        except Exception as e:
            print(f"[{tag}] re-applying locks during followup failed: {e!r}")
        try:
            EnvUtils.set_robot_to_position(self, self.init_robot_base_pos)
        except Exception as e:
            print(f"[{tag}] robot teleport during followup failed: {e!r}")
        self.sim.forward()
        # Settle a few frames so contacts resolve before perception's
        # next tick.
        try:
            zero = np.zeros(self.action_spec[0].shape)
            for _ in range(5):
                self.step(zero)
        except Exception:
            pass
        print(f"[{tag}] chain followup: all cabinets closed, robot "
              f"returned to cab_001 spawn.")

    # ── Success condition ─────────────────────────────────────────────────────

    def _check_success(self):
        """Inert by design. These are scenes, not tasks — there is no
        single goal to check. Per-sub-task success is decided by the eval
        harness's success_check, and the primitives the policy server
        runs use their own atomic task's _check_success
        (robocasa_policy_server.py:1411).
        """
        return False


class MessymemTenCabinets(MessymemUpperRow):
    """10-cabinet row — run with --layout 72."""
    _N_CABINETS = 10


class MessymemFifteenCabinets(MessymemUpperRow):
    """15-cabinet row — run with --layout 73."""
    _N_CABINETS = 15


class MessymemTwentyCabinets(MessymemUpperRow):
    """20-cabinet row — run with --layout 74."""
    _N_CABINETS = 20


# ── Long-horizon fixed scene ──────────────────────────────────────────────────

# Sampling windows, reused from MessymemTwoCabinetsClutterChain where they
# were tuned against the 50-retry placement budget. _WEDGE is the tight
# window used for items pressed against a neighbour; _PIN is tighter still,
# for an item that must land within ~5mm of its x so the pick primitive's
# pre-grasp standoff lines up; _LOOSE is for items whose exact spot doesn't
# matter.
_WEDGE = (0.15, 0.30)
_PIN = (0.12, 0.30)
_LOOSE = (0.20, 0.30)

# Windows for the matching shelves. A 0.70 m usable shelf divides into five
# 0.12 m windows (centres 0.145 m apart) or four 0.16 m ones (0.181 m
# apart); the widest thing placed on either is 0.079 m, so neither spacing
# can collide. Remember that pos is normalised against
# (shelf_width - window_width) / 2 — a WIDER window means LESS spread.
_SHELF = (0.12, 0.28)
_SHELF_WIDE = (0.16, 0.28)


class MessymemLongHorizonTenCabinets(MessymemTenCabinets):
    """Ten-cabinet long-horizon scene — deterministic by category.

    Which item is in which cabinet, where in that cabinet it sits, and
    which cabinets are locked are all fixed: identical on every seed and
    every run. The MESH is not fixed — sample_object() still draws the
    variant from the env RNG (see _PINNED_MODELS), so appearance varies
    while identity and position do not. Items a task depends on for size
    or recognisability are pinned. Sibling of MessymemTenCabinets, which
    keeps the fully randomised behaviour for later use.

    Cabinets are organised by CATEGORY, the way a real kitchen is, so
    instructions can be contextual ("where do the cups go?") rather than
    just nominal ("find object X"):

      cab_001  unlocked  condiments   mustard, KETCHUP (wedged), mayonnaise
      cab_002  unlocked  empty        — (counter: bread, spoon)
      cab_003  LOCKED    baking       flour_bag, sugar_cube, jam
      cab_004  unlocked  pantry       cereal, spaghetti_box, bagged_food, chips
                                        + level1: canned_food, honey_bottle, jam
      cab_005  LOCKED    empty        — (counter: bowl, ladle)
      cab_006  unlocked  dishes       L0: plate, tupperware
                                        + level1: bowl, plate
      cab_007  LOCKED    spices       paprika, cinnamon, turmeric
      cab_008  unlocked  empty        — (counter: mug, whisk)
      cab_009  unlocked  fruits       KETCHUP (alone, far left), lime, apple, banana
      cab_010  unlocked  cups/glasses L0: TEAPOT, mug, coffee_cup, glass_cup
                                        L1: 6 spices

    Three design points worth not undoing:

    1. THE TEAPOT IS IN THE LAST CABINET. It is the only teapot in the
       room and nothing else resembles one, so sub-task 1 ("find the
       teapot") cannot terminate early on a lookalike — the robot has to
       work the row all the way to cab_010, which is what seeds memory of
       every other cabinet, including which ones are locked and which are
       empty. Every later sub-task depends on that sweep having happened.

    2. THE TWO KETCHUPS ARE MAXIMALLY SEPARATED AND SPATIALLY OPPOSITE.
       cab_001's ketchup is WEDGED between the mustard and the mayonnaise;
       cab_009's sits ALONE at the far left with the fruit packed to its
       right. That single arrangement carries two different evals at once:
         - contextual disambiguation — "the ketchup with the mustard and
           mayonnaise" vs "the ketchup next to the lime and apple" resolve
           to different cabinets 8 apart, so a planner that remembers only
           "ketchup is in a cabinet" cannot answer either.
         - clutter-spatial retrieval — a pick instruction must choose
           cab_009, and only keyframes carry the wedged/accessible signal
           (both cabinets' contents lists say "ketchup").

    3. CONTENTS AND COUNTER ARE ANTI-CORRELATED. Every cabinet that is
       locked or empty has items on the counter below it; every unlocked
       cabinet with contents has a bare counter. A planner cannot use
       "stuff on the counter" as a proxy for "stuff in the cabinet" — if
       anything the correlation runs backwards.

    Locked cabinets never open, so cab_003 / cab_007's contents are ground
    truth only and are never observable. That is intentional: they exist
    so an instruction can target something that is genuinely unreachable
    (absence-with-a-reason) as distinct from something that is simply not
    in the room at all. ``unlock_cabinet`` can release one mid-chain via
    an instruction's setup.env_method, but nothing does so by default.
    """

    _LOCKED_CABS = ("cab_003", "cab_005", "cab_007")

    # Only the two ketchups need to survive the graspable filter — they are
    # the sole pick targets. Everything else is find/route-only, and
    # leaving the filter off avoids emptying a category at build time.
    _GRASPABLE = {"ketchup"}

    # ketchup_7 is the thin 4.4 cm bottle MessymemTwoCabinetsClutterChain
    # pins for the same reason: the pick primitive's pre-grasp standoff
    # cannot accommodate the chunkier variants the category sampler
    # otherwise returns. Both ketchups use it, so sub-task 16's choice
    # between them is about position, never about which bottle is easier
    # to grasp. Mustard001 is the iconic yellow bottle with a dark blue
    # label — unmistakable against the red ketchup beside it, which is
    # what makes "the ketchup stored with the mustard" answerable from a
    # keyframe rather than from a contents list.
    # EVERY item is pinned, which makes this scene fully deterministic —
    # same models, same sizes, same appearance on every seed and every run —
    # and removes the placement fragility that unpinned categories cause.
    #
    # Variants were chosen by measuring the footprint of every model in each
    # category and taking the smallest, because robocasa's categories span
    # wildly different real-world objects: "plate" ranges 0.28-0.49 m (side
    # plate to serving platter), "tupperware" 0.34-0.64, "knife" 0.28-0.51.
    # An unpinned draw could pick the 0.49 m platter, which does not fit a
    # 0.74 x 0.33 m shelf, and because draws are non-local that failure
    # would appear to come from an unrelated cabinet.
    #
    # ketchup and mustard are the two exceptions to "smallest": they are
    # pinned to variants proven in MessymemTwoCabinetsClutterChain —
    # ketchup_7 because the pick primitive's pre-grasp standoff needs the
    # thin 4.4 cm bottle, Mustard001 because its yellow body and blue label
    # are unmistakable against the red ketchup beside it.
    _PINNED_MODELS = {
        "ketchup": "objects/objaverse/ketchup/ketchup_7/model.xml",
        "mustard": "objects/lightwheel/mustard/Mustard001/model.xml",
        "mayonnaise": "objects/lightwheel/mayonnaise/Mayonnaise006/model.xml",
        "flour_bag": "objects/lightwheel/flour_bag/FlourBag009/model.xml",
        "sugar_cube": "objects/lightwheel/sugar_cube/SugarCube001/model.xml",
        "jam": "objects/aigen_objs/jam/jam_14/model.xml",
        "cereal": "objects/aigen_objs/cereal/cereal_4/model.xml",
        "spaghetti_box": "objects/aigen_objs/spaghetti_box/spaghetti_box_9/model.xml",
        "bagged_food": "objects/aigen_objs/bagged_food/bagged_food_6/model.xml",
        "chips": "objects/aigen_objs/chips/chips_6/model.xml",
        "canned_food": "objects/aigen_objs/canned_food/canned_food_6/model.xml",
        "honey_bottle": "objects/aigen_objs/honey_bottle/honey_bottle_2/model.xml",
        "plate": "objects/aigen_objs/plate/plate_5/model.xml",          # 0.28 m side plate
        "tupperware": "objects/lightwheel/tupperware/Tupperware039/model.xml",
        "jar": "objects/lightwheel/jar/Jar007/model.xml",
        "bowl": "objects/objaverse/bowl/bowl_14/model.xml",             # 0.23 m bowl
        "paprika": "objects/lightwheel/paprika/Paprika002/model.xml",
        "cinnamon": "objects/lightwheel/cinnamon/Cinnamon002/model.xml",
        "turmeric": "objects/lightwheel/turmeric/Turmeric006/model.xml",
        "lime": "objects/aigen_objs/lime/lime_5/model.xml",
        "apple": "objects/objaverse/apple/apple_15/model.xml",
        "banana": "objects/aigen_objs/banana/banana_8/model.xml",
        "teapot": "objects/objaverse/teapot/teapot_13/model.xml",
        "mug": "objects/aigen_objs/mug/mug_7/model.xml",
        "coffee_cup": "objects/aigen_objs/coffee_cup/coffee_cup_2/model.xml",
        "glass_cup": "objects/lightwheel/glass_cup/GlassCup025/model.xml",
        "bread": "objects/aigen_objs/bread/bread_5/model.xml",
        "spoon": "objects/aigen_objs/spoon/spoon_2/model.xml",
        "orange": "objects/aigen_objs/orange/orange_1/model.xml",
        "fork": "objects/aigen_objs/fork/fork_3/model.xml",
        "knife": "objects/aigen_objs/knife/knife_5/model.xml",
        "avocado": "objects/objaverse/avocado/avocado_8/model.xml",
        "ladle": "objects/aigen_objs/ladle/ladle_0/model.xml",
        "croissant": "objects/aigen_objs/croissant/croissant_1/model.xml",
        "pineapple": "objects/aigen_objs/pineapple/pineapple_2/model.xml",
        "tomato": "objects/aigen_objs/tomato/tomato_1/model.xml",
        "whisk": "objects/aigen_objs/whisk/whisk_5/model.xml",
        "shaker": "objects/aigen_objs/shaker/shaker_3/model.xml",
        "syrup_bottle": "objects/aigen_objs/syrup_bottle/syrup_bottle_8/model.xml",
        "salt_and_pepper_shaker": "objects/lightwheel/salt_and_pepper_shaker/PepperShaker006/model.xml",
        "oil_and_vinegar_bottle": "objects/lightwheel/oil_and_vinegar_bottle/OilBottle005/model.xml",

        # ── object_matching triples ───────────────────────────────────────
        # Seven categories, three unmistakably different variants each,
        # spread over three cabinets so that neither the category name nor
        # the cabinet's theme identifies which one is meant. See
        # README_object_matching.md.
        #
        # Two kinds of discriminator, and the difference is load-bearing:
        #
        #   COLOUR survives into the scene graph. The analyzer writes a
        #   `description` per item and those descriptions really do say
        #   "yellow squeeze bottle" / "white jar with a blue lid".
        #
        #   BRANDING does not. Across a whole 10-cabinet run the analyzer
        #   never once wrote a brand name — a Campbell's tin came back as
        #   "small silver cylindrical can".
        #
        # So a two-variant colour split would leak: "the red can" answers
        # itself off the contents list. The THIRD variant is what closes
        # that — with a plain red can in the room as well as the Coke, the
        # logo is the only thing that separates them, and the logo lives
        # only in the keyframes. Do not "simplify" a triple back to a pair.
        "can#coke":            "objects/objaverse/can/can_15/model.xml",
        "can#pepsi":           "objects/objaverse/can/can_17/model.xml",
        "can#plain":           "objects/objaverse/can/can_16/model.xml",

        "coffee_cup#starbucks": "objects/objaverse/coffee_cup/coffee_cup_9/model.xml",
        "coffee_cup#kraft":     "objects/objaverse/coffee_cup/coffee_cup_14/model.xml",
        "coffee_cup#plain":     "objects/objaverse/coffee_cup/coffee_cup_16/model.xml",

        # Cereal is the one class whose identity is printed on a single
        # face, hence the pinned rotation on every cereal placement below.
        "cereal#corn_flakes":  "objects/objaverse/cereal/cereal_11/model.xml",
        "cereal#fruit_loops":  "objects/objaverse/cereal/cereal_7/model.xml",
        "cereal#trix":         "objects/objaverse/cereal/cereal_12/model.xml",

        # All three mustards are yellow on purpose: the contents list reads
        # "yellow mustard bottle" three times and carries no signal at all.
        "mustard#labelled":    "objects/lightwheel/mustard/Mustard001/model.xml",
        "mustard#plain":       "objects/lightwheel/mustard/Mustard003/model.xml",
        "mustard#graphic":     "objects/lightwheel/mustard/Mustard008/model.xml",

        "mug#red":             "objects/objaverse/mug/mug_13/model.xml",
        "mug#teal":            "objects/objaverse/mug/mug_15/model.xml",
        "mug#yellow":          "objects/objaverse/mug/mug_11/model.xml",

        "juice#grape":         "objects/lightwheel/juice/Juice012/model.xml",
        "juice#orange":        "objects/lightwheel/juice/Juice007/model.xml",
        "juice#cranberry":     "objects/lightwheel/juice/Juice006/model.xml",

        # Reusable flask / disposable PET / squeeze sports bottle — three
        # different *kinds* of water bottle, so "the reusable one" and "the
        # recyclable one" are separate questions off one triple. The
        # disposable one is the only aigen model in the set; the SG label
        # is "water bottle" for all three, so the registry does not leak.
        "water_bottle#reusable":   "objects/objaverse/water_bottle/water_bottle_21/model.xml",
        "water_bottle#disposable": "objects/aigen_objs/water_bottle/water_bottle_4/model.xml",
        "water_bottle#sports":     "objects/objaverse/water_bottle/water_bottle_22/model.xml",
    }

    # Yaw for the cereal boxes, in radians, measured in the placement
    # sampler's frame. 0.0 leaves the box at the sampler's reference
    # orientation. If the boxes come up showing their nutrition panels,
    # nudge this by pi (or +-pi/2) — one number, every cereal follows it.
    _CEREAL_YAW = 0.0

    _FIXED_CONTENTS = {
        # ── condiments — ketchup WEDGED between its two neighbours ────────
        "cab_001": [
            ("mustard#labelled", (-0.35, 0.0), _WEDGE),
            ("ketchup",          ( 0.00, 0.0), _PIN),
            ("mayonnaise",       ( 0.35, 0.0), _WEDGE),
            # ── matching shelf ──────────────────────────────────────────
            # Four of the seven triples put a leg here. This shelf was
            # empty, which is why it absorbs the most; level0 above is
            # untouched so the cabinet still reads as "condiments" for the
            # put-away sub-tasks. 4 x 0.16 m = 0.64 of a 0.70 m shelf.
            ("can#coke",          (-1.00, 0.0), _SHELF_WIDE, "level1"),
            ("cereal#fruit_loops", (-0.33, 0.0), _SHELF_WIDE, "level1",
             _CEREAL_YAW),
            # `y` IS NOT A DEPTH KNOB HERE — measured, do not retry it.
            # This mug sits 0.22 m past the cabinet's front face, which is
            # a 0.79 m pre-grasp against a 0.85 m limit and the tightest
            # margin of any target we use. The obvious fix is to pull it
            # forward with y, and it does not work: y = 0.0 gives depth
            # 0.222 and y = +1.0 gives 0.227 — five millimetres, and the
            # wrong way. Shrinking the window to (0.16, 0.10) to pin it
            # harder makes placement infeasible outright (50 failed
            # _load_model attempts). The sampler clips to the shelf's own
            # placement region, and that region is what sets the depth.
            # Anything on this shelf will sit about this far back.
            ("mug#yellow",        ( 0.33, 0.0), _SHELF_WIDE, "level1"),
            ("juice#grape",       ( 1.00, 0.0), _SHELF_WIDE, "level1"),
        ],
        "cab_002": [],
        # ── baking — LOCKED, ground truth only, never observable ──────────
        "cab_003": [
            ("flour_bag",  (-0.40, 0.0), _WEDGE),
            ("sugar_cube", ( 0.00, 0.0), _WEDGE),
            ("jam",        ( 0.40, 0.0), _WEDGE),
        ],
        # ── pantry / dry goods — stocked on TWO shelves ───────────────────
        # level0 is the bottom shelf (z ~ 1.42), level1 the middle one
        # (z ~ 1.72). level2 (z ~ 2.01) is deliberately left empty: see the
        # shelf note in _get_obj_cfgs_fixed.
        "cab_004": [
            # The pantry's own cereal is the Trix — the DECOY for
            # match_cereal. Its target (Corn Flakes) lives in cab_009, so a
            # planner that reasons "cereal belongs in the pantry" is wrong.
            ("cereal#trix",   (-0.60, 0.0), _LOOSE, None, _CEREAL_YAW),
            ("spaghetti_box", (-0.20, 0.0), _LOOSE),
            ("bagged_food",   ( 0.20, 0.0), _LOOSE),
            ("chips",         ( 0.60, 0.0), _LOOSE),
            # Level1 goes from 3 items to 5, so the windows narrow from
            # 0.16 to 0.12 and all five re-space evenly. Widest occupant is
            # the reusable bottle at 0.070 m; spacing is 0.145 m.
            ("canned_food",             (-1.00, 0.0), _SHELF, "level1"),
            ("honey_bottle",            (-0.50, 0.0), _SHELF, "level1"),
            ("jam",                     ( 0.00, 0.0), _SHELF, "level1"),
            ("coffee_cup#starbucks",    ( 0.50, 0.0), _SHELF, "level1"),
            ("water_bottle#reusable",   ( 1.00, 0.0), _SHELF, "level1"),
        ],
        "cab_005": [],
        # ── dishes & food containers ──────────────────────────────────────
        # Sized empirically, not by eye. robocasa's plate and bowl render
        # at ~0.5 m across, so TWO of them cannot share a 0.8 m cabinet,
        # and tray / saucepan do not fit at all at any window size. The
        # windows below are the ones this trio was measured to pack into:
        # plate 0.47, tupperware 0.27, jar 0.20. Do not shrink them.
        # Two items per shelf, not three. The shelf is 0.74 m wide (0.70
        # after margin) and these are the widest items in the kitchen:
        # plate 0.28, tupperware 0.34, bowl 0.23. Three of them on one
        # shelf needs 0.83 m and fails — the jar that used to sit here was
        # dropped rather than shrinking the dishes, since the point of
        # this cabinet is that it holds recognisable dinnerware.
        "cab_006": [
            ("plate",      (-0.50, 0.0), (0.30, 0.32)),
            ("tupperware", ( 0.50, 0.0), (0.36, 0.30)),
            # Middle shelf: a side plate and a bowl, both pinned small
            # (0.28 / 0.20 m). Nothing large goes up here — a cutting
            # board or serving platter is wider than the shelf is deep.
            # pos is normalised against (shelf_width - window_width)/2, NOT
            # against the shelf. With a 0.30 m window on a 0.70 m shelf the
            # usable half-span is only 0.20 m, so pos=+-0.45 puts these two
            # centres 0.18 m apart — closer than their combined radii, and
            # the sampler rejects the overlap. Pinning to the extremes gives
            # 0.43 m of separation. Widen a window and this spread SHRINKS.
            # The bowl keeps its end of the shelf. The second plate that
            # used to sit at the other end is gone: cab_006 already reads
            # as dinnerware from level0 and from this bowl, and the plate
            # was a duplicate that ground_truth_contents deduplicated away
            # anyway. Dropping it frees 0.28 m — the only slack in the
            # whole budget, and it goes to three matching legs.
            ("bowl",                     (-1.00, 0.0), (0.24, 0.28), "level1"),
            ("mustard#plain",            (-0.10, 0.0), _SHELF, "level1"),
            ("mug#red",                  ( 0.45, 0.0), _SHELF, "level1"),
            ("water_bottle#disposable",  ( 1.00, 0.0), _SHELF, "level1"),
        ],
        # ── spices — LOCKED, ground truth only ────────────────────────────
        "cab_007": [
            ("paprika",  (-0.40, 0.0), _WEDGE),
            ("cinnamon", ( 0.00, 0.0), _WEDGE),
            ("turmeric", ( 0.40, 0.0), _WEDGE),
        ],
        "cab_008": [],
        # ── fruits — ketchup ALONE on the far left, fruit packed right ────
        # Mirrors cab_1 in MessymemTwoCabinetsClutterChain: the pickable
        # bottle is pinned hard left with clear side access, everything
        # else is clustered out of the gripper's path.
        "cab_009": [
            ("ketchup", (-0.85, -0.40), _PIN),
            ("lime",    ( 0.15,  0.00), _WEDGE),
            ("apple",   ( 0.50,  0.00), _WEDGE),
            ("banana",  ( 0.85,  0.00), _WEDGE),
            # ── matching shelf ──────────────────────────────────────────
            # The other empty level1, and the home of three targets. The
            # Corn Flakes being HERE rather than in the pantry is the whole
            # point of match_cereal_trix's sibling; the orange juice being
            # here is the decoy for match_juice_grape, since fruit is
            # exactly where a category-reasoning planner would look.
            # Level0 is untouched — the ketchup keeps its clear side
            # access for pick_ketchup_accessible.
            ("coffee_cup#plain",    (-1.00, 0.0), _SHELF, "level1"),
            ("cereal#corn_flakes",  (-0.50, 0.0), _SHELF, "level1",
             _CEREAL_YAW),
            ("mustard#graphic",     ( 0.00, 0.0), _SHELF, "level1"),
            ("juice#orange",        ( 0.50, 0.0), _SHELF, "level1"),
            ("can#pepsi",           ( 1.00, 0.0), _SHELF, "level1"),
        ],
        # ── cups & glasses — holds the teapot, sub-task 1's target ────────
        "cab_010": [
            ("teapot",              (-0.55, 0.0), _LOOSE),
            # The cups cabinet is the OBVIOUS home for a mug and a coffee
            # cup, so both of its cups are decoys — the matching targets
            # are in cab_006 and cab_004. Same models as before in every
            # respect except which variant they point at.
            ("mug#teal",            (-0.10, 0.0), _LOOSE),
            ("coffee_cup#kraft",    ( 0.35, 0.0), _LOOSE),
            ("glass_cup",           ( 0.80, 0.0), _LOOSE),
            # Spice rack on the shelf above the cups. Six items across a
            # 0.70 m usable shelf: with a 0.12 m window the usable half-span
            # is 0.29 m, so evenly spacing pos over [-1, 1] leaves 0.116 m
            # between centres. Widest item (turmeric, 0.131) goes on an end
            # so no adjacent pair exceeds that gap.
            #
            # paprika is deliberately NOT here — it is the one spice that
            # exists only inside locked cab_007, which is what makes
            # find_paprika_unreachable a genuine "cannot be reached"
            # sub-task rather than a findable one.
            # Three of the six spices made way for the third leg of three
            # triples, at the SAME positions and window sizes — this shelf
            # already packed, so reusing its geometry keeps a known-good
            # arrangement. Dropped: salt_and_pepper_shaker and shaker (both
            # still on cab_007's counter, so still in the room) and
            # syrup_bottle (leaves the scene; nothing referenced it).
            # cinnamon STAYS: paprika has to remain the one spice that
            # exists only inside a locked cabinet, which is what makes
            # find_paprika_unreachable unreachable-for-a-reason.
            ("turmeric",               (-1.00, 0.0), (0.12, 0.26), "level1"),
            ("oil_and_vinegar_bottle", (-0.60, 0.0), (0.12, 0.26), "level1"),
            ("can#plain",              (-0.20, 0.0), (0.12, 0.26), "level1"),
            ("juice#cranberry",        ( 0.20, 0.0), (0.12, 0.26), "level1"),
            ("cinnamon",               ( 0.60, 0.0), (0.12, 0.26), "level1"),
            ("water_bottle#sports",    ( 1.00, 0.0), (0.12, 0.26), "level1"),
        ],
    }

    # Counter items sit ONLY below locked or empty cabinets. Each one is a
    # deliberate "put-away" probe: bread and chips-adjacent goods belong
    # with the pantry (cab_004), bowl and ladle with the kitchenware
    # (cab_006), mug with the cups (cab_010). Two of them — bowl and mug —
    # duplicate an item in their destination cabinet, so the destination is
    # recoverable by exact match; the rest (bread, spoon, ladle, whisk,
    # croissant, fork, cutting_board) have no duplicate and require
    # reasoning about what the cabinet is FOR.
    # Three of these are MATCHING PROBES — the object a counter-referencing
    # instruction points at ("the mug on the counter next to the bread").
    # They are the reason those sub-tasks need keyframes at all: a counter
    # object's scene-graph node is `label` + `pos` and nothing else, so its
    # appearance exists only in the frames captured during the sweep.
    #
    # Each probe is pinned to the same model as its target in a cabinet, and
    # sits under a DIFFERENT cabinet from that target so proximity gives
    # nothing away. The instruction names a neighbour ("next to the bread")
    # rather than the attribute — saying "the red mug" would hand the answer
    # to a planner that never looked.
    #
    # cab_008's mug is deliberately left untagged and unchanged: it belongs
    # to put_mug_away, which routes it to cab_010 by CATEGORY. match_mug
    # routes a different mug to cab_006 by APPEARANCE, so the two must not
    # share an object or they contradict each other.
    _FIXED_COUNTER = {
        "cab_002": ["bread", "spoon", "orange", "mug#red"],
        "cab_003": ["fork", "knife", "avocado", "mustard#plain"],
        "cab_005": ["bowl", "ladle", "plate", "coffee_cup#starbucks"],
        "cab_007": ["croissant", "pineapple", "salt_and_pepper_shaker", "shaker"],
        "cab_008": ["mug", "whisk", "tomato"],
    }

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Find the teapot in the upper cabinets."
        # Category label per cabinet — ground truth for the "where does
        # this belong?" instructions. Not shown to the planner.
        ep_meta["cabinet_categories"] = {
            "cab_001": "condiments",
            "cab_002": "empty",
            "cab_003": "baking (locked)",
            "cab_004": "pantry / dry goods",
            "cab_005": "empty (locked)",
            "cab_006": "dishes & food containers",
            "cab_007": "spices (locked)",
            "cab_008": "empty",
            "cab_009": "fruits",
            "cab_010": "cups & glasses",
        }
        return ep_meta
