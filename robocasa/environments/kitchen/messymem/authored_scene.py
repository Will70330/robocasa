"""
authored_scene.py — pin objects to exact world poses.

The robocasa placement sampler is built to RANDOMISE: you hand it a
fixture and a window and it draws a spot inside. Scenes authored by hand
in the layout builder need the opposite — this object, at this point, at
this yaw, every reset. ``abs_object_cfg`` is that, expressed in the
sampler's own vocabulary via ``sample_args.reference``, which accepts a
world (x, y, z) directly (placement_samplers.py:348).

``AuthoredScene`` is the mixin an env subclasses to replay a session file
written by scripts/sim/layout_builder.py.

Two landmines, both of which fail far from their cause:

  * ``reference`` MUST be a tuple. kitchen.py:1027 does ``ref in
    sliding_fixture_objs`` — a dict membership test — so a list raises
    ``TypeError: unhashable type: 'list'`` from deep inside _load_model.
    json.loads hands back lists, so every load boundary coerces.

  * ``ensure_valid_placement`` must be off. The window here is degenerate
    (0 x 0), so the validity check rejects it, and robocasa's 50-attempt
    retry loop re-runs an IDENTICAL function — 50 identical failures and
    a RuntimeError. Same trap the _FIXED_CONTENTS docstring warns about
    in messymem_n_uppers.py.
"""
import json
import os

import numpy as np

import robocasa
from robocasa.utils import object_utils as OU

SCHEMA = 1


def abs_object_cfg(name, mjcf_path, xyz, yaw=0.0, graspable=None):
    """Object cfg landing *name* at world *xyz* with world *yaw*.

    *xyz* is the point the object's BOTTOM rests on — ``on_top=True`` makes
    the sampler subtract the mesh's bottom_offset — so it is the point you
    clicked, not an opaque body origin.
    """
    cfg = dict(
        name=name,
        type="object",
        obj_groups=mjcf_path,
        placement=dict(
            size=(0.0, 0.0),
            pos=(0.0, 0.0),
            rotation=float(yaw),
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            sample_args=dict(
                reference=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                on_top=True,
            ),
        ),
    )
    if graspable is not None:
        cfg["graspable"] = bool(graspable)
    return cfg


def resolve_session_path(path):
    """Absolute path for a session file.

    Scene configs live in the messymem repo (configs/sim/scene_objects/),
    not in robocasa, so a relative path is resolved against $MESSYMEM_ROOT
    when set and otherwise against the repo this submodule sits inside.
    """
    if os.path.isabs(path):
        return path
    root = os.environ.get("MESSYMEM_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(robocasa.__file__)))
    return os.path.join(root, path)


def load_session(path):
    path = resolve_session_path(path)
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    return doc


def save_session(path, base, entries):
    doc = dict(
        schema=SCHEMA,
        env=dict(
            task=type(base).__name__,
            layout_id=int(base.layout_id),
            style_id=int(base.style_id),
            obj_registries=list(base.obj_registries),
        ),
        robot=dict(
            init_robot_base_pos=[float(v) for v in base.init_robot_base_pos],
            init_robot_base_ori=[float(v) for v in base.init_robot_base_ori],
        ),
        objects=entries,
    )
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return doc


def entry_pos(base, entry):
    """World point for *entry*, preferring its fixture anchor.

    Fixture positions are deterministic given (layout, style), so the
    absolute pose alone would do — but anchoring means a layout survives a
    fixture being nudged, which happens while a scene is being designed.
    """
    anchor = entry.get("anchor")
    if anchor and anchor.get("fixture") in base.fixtures:
        return OU.get_pos_after_rel_offset(
            base.fixtures[anchor["fixture"]],
            np.asarray(anchor["rel_offset"], dtype=float),
        )
    return np.asarray(entry["pos"], dtype=float)


def cfgs_from_entries(base, entries):
    """Session entries -> object cfgs, resolving anchors against *base*."""
    out = []
    for e in entries:
        rel = e["mjcf_path"]
        path = rel if os.path.isabs(rel) else os.path.join(
            robocasa.models.assets_root, rel)
        out.append(abs_object_cfg(
            e["name"], path, entry_pos(base, e), float(e.get("yaw", 0.0)),
            graspable=e.get("graspable")))
    return out


class AuthoredScene:
    """Mixin: replay a layout-builder session as the scene's objects.

    Point ``_SESSION_FILE`` at a session json, or set ``_PINNED_PLACEMENTS``
    inline as ``{name: (mjcf_rel_path, (x, y, z), yaw)}``. The session file
    wins when both are set.
    """

    _SESSION_FILE = None
    _PINNED_PLACEMENTS = {}
    _GRASPABLE = set()

    def _authored_entries(self):
        if self._SESSION_FILE:
            return load_session(self._SESSION_FILE)["objects"]
        return [
            dict(name=name, mjcf_path=rel, pos=list(pos), yaw=float(yaw),
                 graspable=name in self._GRASPABLE)
            for name, (rel, pos, yaw) in self._PINNED_PLACEMENTS.items()
        ]

    def _get_obj_cfgs(self):
        return cfgs_from_entries(self, self._authored_entries())
