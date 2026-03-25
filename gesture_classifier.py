"""Rule-based hand gesture classifier using normalized 2-D landmarks.

Gestures
--------
open   — all four fingers extended
fist   — all four fingers curled
pinch  — thumb tip and index tip close together
point  — index extended, remaining fingers curled

Features
--------
pinch_distance  : Euclidean distance between thumb tip (4) and index tip (8),
                  in the same normalized [0, 1] coordinate space as the landmarks.
flexion_angles  : Per-finger bend angle at the proximal interphalangeal joint,
                  in degrees.  0° = straight, ~90°+ = fully curled.
                  Order: [thumb, index, middle, ring, pinky]
"""

import math

# Landmark triplets (base, joint, distal) for computing the flexion angle at 'joint'.
# The angle between (base → joint) and (joint → distal) vectors gives the bend.
# When the finger is straight both vectors are antiparallel → 180°, so:
#   flexion = 180° − raw_angle  (0° = straight, grows as the finger curls)
_FINGER_TRIPLETS: list[tuple[int, int, int]] = [
    (1, 2, 3),    # Thumb  : CMC → MCP → IP
    (5, 6, 7),    # Index  : MCP → PIP → DIP
    (9, 10, 11),  # Middle : MCP → PIP → DIP
    (13, 14, 15), # Ring   : MCP → PIP → DIP
    (17, 18, 19), # Pinky  : MCP → PIP → DIP
]

_THUMB_TIP = 4
_INDEX_TIP = 8

# Distance below which thumb ↔ index is considered a pinch.
_PINCH_THRESHOLD = 0.08

# Flexion angle (degrees) above which a finger is considered "curled".
_FLEX_CURLED = 70.0


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _vec2(a: dict, b: dict) -> tuple[float, float]:
    return a["x"] - b["x"], a["y"] - b["y"]


def _angle_at_vertex(a: dict, b: dict, c: dict) -> float:
    """Angle at b formed by the path a–b–c, in degrees [0, 180]."""
    v1 = _vec2(a, b)
    v2 = _vec2(c, b)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


# ---------------------------------------------------------------------------
# Public feature extractors
# ---------------------------------------------------------------------------

def compute_pinch_distance(landmarks: list[dict]) -> float:
    """Euclidean distance between thumb tip (4) and index tip (8)."""
    t = landmarks[_THUMB_TIP]
    i = landmarks[_INDEX_TIP]
    return math.hypot(t["x"] - i["x"], t["y"] - i["y"])


def compute_flexion_angles(landmarks: list[dict]) -> list[float]:
    """Per-finger flexion angle in degrees.

    Returns a list of 5 values [thumb, index, middle, ring, pinky].
    0° = fully extended, larger values = more curled.
    """
    angles: list[float] = []
    for base_idx, joint_idx, distal_idx in _FINGER_TRIPLETS:
        raw = _angle_at_vertex(
            landmarks[base_idx],
            landmarks[joint_idx],
            landmarks[distal_idx],
        )
        # Straight finger → raw ≈ 180° → flexion ≈ 0°
        angles.append(max(0.0, 180.0 - raw))
    return angles


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _flex_score(angle: float) -> float:
    """Continuous score: 0 = extended, 1 = fully curled."""
    return min(1.0, max(0.0, angle / _FLEX_CURLED))


def classify_gesture(
    landmarks: list[dict],
) -> tuple[str, float, float, list[float]]:
    """Classify hand gesture from 21 MediaPipe landmarks.

    Parameters
    ----------
    landmarks:
        List of 21 dicts, each with keys ``id``, ``x``, ``y``.

    Returns
    -------
    label : str
        One of ``"open"``, ``"fist"``, ``"pinch"``, ``"point"``.
    confidence : float
        Score in [0, 1] indicating how strongly the gesture was detected.
    pinch_distance : float
        Raw distance between thumb tip and index tip.
    flexion_angles : list[float]
        Per-finger flexion in degrees [thumb, index, middle, ring, pinky].
    """
    pinch_dist = compute_pinch_distance(landmarks)
    flexions = compute_flexion_angles(landmarks)

    _, idx_f, mid_f, ring_f, pinky_f = flexions

    # --- pinch ---------------------------------------------------------------
    # Strong when thumb and index tips are close together.
    pinch_score = max(0.0, 1.0 - pinch_dist / _PINCH_THRESHOLD)

    # --- point ---------------------------------------------------------------
    # Index extended; middle, ring, pinky curled.
    idx_extended = 1.0 - _flex_score(idx_f)
    others_curled = (_flex_score(mid_f) + _flex_score(ring_f) + _flex_score(pinky_f)) / 3.0
    point_score = (idx_extended + others_curled) / 2.0

    # --- fist ----------------------------------------------------------------
    # All four fingers (index … pinky) curled.
    fist_score = (
        _flex_score(idx_f) + _flex_score(mid_f) +
        _flex_score(ring_f) + _flex_score(pinky_f)
    ) / 4.0

    # --- open ----------------------------------------------------------------
    # All four fingers extended.
    open_score = (
        (1.0 - _flex_score(idx_f)) + (1.0 - _flex_score(mid_f)) +
        (1.0 - _flex_score(ring_f)) + (1.0 - _flex_score(pinky_f))
    ) / 4.0

    scores: dict[str, float] = {
        "pinch": pinch_score,
        "point": point_score,
        "fist":  fist_score,
        "open":  open_score,
    }
    label = max(scores, key=scores.__getitem__)
    confidence = round(scores[label], 4)

    return label, confidence, round(pinch_dist, 4), [round(a, 2) for a in flexions]
