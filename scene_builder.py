"""Build foxglove.SceneUpdate messages from hand landmark data.

Each detected hand becomes one SceneEntity containing:
  - 21 sphere primitives (one per landmark), colored by finger
  - 6 line-list primitives (one per finger + palm group), colored by finger

Landmark coordinates are normalized [0, 1] with y=0 at the top of the image.
We center and flip-y when mapping to 3-D scene space so the hand appears
right-side-up in Foxglove's 3D panel.
"""

from __future__ import annotations

from tracker import HandLandmarks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCALE = 1.0          # scene units per normalized unit
_SPHERE_SIZE = 0.018  # scene units
_LINE_THICKNESS = 0.006

# Foxglove LinePrimitive type field: 2 = LINE_LIST (each consecutive pair of
# points forms one independent segment, so no "caps" between segments)
_LINE_LIST = 2

_IDENTITY_QUAT = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
_ORIGIN_POS    = {"x": 0.0, "y": 0.0, "z": 0.0}
_ORIGIN_POSE   = {"position": _ORIGIN_POS, "orientation": _IDENTITY_QUAT}

# Per-finger RGBA colors (floats in [0, 1])
_COLORS: dict[str, dict] = {
    "wrist":  {"r": 0.72, "g": 0.72, "b": 0.72, "a": 1.0},  # landmark 0 finger label
    "thumb":  {"r": 1.00, "g": 0.22, "b": 0.22, "a": 1.0},
    "index":  {"r": 1.00, "g": 0.60, "b": 0.00, "a": 1.0},
    "middle": {"r": 0.18, "g": 0.88, "b": 0.18, "a": 1.0},
    "ring":   {"r": 0.22, "g": 0.45, "b": 1.00, "a": 1.0},
    "pinky":  {"r": 0.78, "g": 0.22, "b": 1.00, "a": 1.0},
    "palm":   {"r": 0.72, "g": 0.72, "b": 0.72, "a": 1.0},  # palm cross-connections
}

# Skeleton connections grouped by finger (LINE_LIST pairs)
_BONE_GROUPS: dict[str, list[tuple[int, int]]] = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring":   [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
    "palm":   [(5, 9), (9, 13), (13, 17)],
}

# Entity lifetime: after this duration with no update the entity auto-expires.
_LIFETIME_NS = 150_000_000  # 150 ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_3d(lm: dict) -> dict:
    """Map a normalized landmark to a 3-D scene position."""
    return {
        "x": (lm["x"] - 0.5) * _SCALE,
        "y": -(lm["y"] - 0.5) * _SCALE,  # flip: image y=0 → scene y=+0.5
        "z": -lm["z"] * _SCALE,           # MediaPipe z: negative toward camera → positive out
    }


def _sphere(pos: dict, color: dict) -> dict:
    return {
        "pose": {"position": pos, "orientation": _IDENTITY_QUAT},
        "size": {"x": _SPHERE_SIZE, "y": _SPHERE_SIZE, "z": _SPHERE_SIZE},
        "color": color,
    }


def _line_list(points: list[dict], color: dict) -> dict:
    return {
        "type": _LINE_LIST,
        "pose": _ORIGIN_POSE,
        "thickness": _LINE_THICKNESS,
        "scale_invariant": False,
        "points": points,
        "color": color,
        "colors": [],
        "indices": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_entity(hand: HandLandmarks, timestamp_ns: int) -> dict:
    positions = [_to_3d(lm) for lm in hand.landmarks]

    spheres = [
        _sphere(positions[i], _COLORS[lm["finger"]])
        for i, lm in enumerate(hand.landmarks)
    ]

    lines = []
    for finger, bones in _BONE_GROUPS.items():
        pts: list[dict] = []
        for a, b in bones:
            pts.append(positions[a])
            pts.append(positions[b])
        lines.append(_line_list(pts, _COLORS[finger]))

    sec  = timestamp_ns // 1_000_000_000
    nsec = timestamp_ns %  1_000_000_000

    return {
        "timestamp":    {"sec": sec, "nsec": nsec},
        "frame_id":     "hand",
        "id":           f"hand_{hand.hand_label.lower()}",
        "lifetime":     {"sec": 0, "nsec": _LIFETIME_NS},
        "frame_locked": False,
        "metadata":     [],
        "arrows":       [],
        "cubes":        [],
        "spheres":      spheres,
        "lines":        lines,
        "triangles":    [],
        "texts":        [],
        "models":       [],
    }


def build_scene_update(landmarks_list: list[HandLandmarks], timestamp_ns: int) -> dict:
    """Return a foxglove.SceneUpdate dict ready for JSON serialization."""
    return {
        "deletions": [],
        "entities":  [_build_entity(h, timestamp_ns) for h in landmarks_list],
    }
