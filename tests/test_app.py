"""
Tests for the motion-tracker application.

Run with:
    python -m pytest tests/ -v
    python -m unittest discover tests/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import websockets

# Make project root importable when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_classifier import classify_gesture, compute_flexion_angles, compute_pinch_distance
from scene_builder import build_scene_update
from schemas import (
    COMPRESSED_IMAGE_SCHEMA,
    HAND_GESTURE_SCHEMA,
    HAND_LANDMARKS_SCHEMA,
    SCENE_UPDATE_SCHEMA,
)
from tracker import HandLandmarks

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# (finger, joint) label for each of the 21 MediaPipe landmarks, in index order.
_LM_LABELS = [
    ("wrist",  "wrist"),
    ("thumb",  "cmc"),  ("thumb",  "mcp"),  ("thumb",  "ip"),   ("thumb",  "tip"),
    ("index",  "mcp"),  ("index",  "pip"),  ("index",  "dip"),  ("index",  "tip"),
    ("middle", "mcp"),  ("middle", "pip"),  ("middle", "dip"),  ("middle", "tip"),
    ("ring",   "mcp"),  ("ring",   "pip"),  ("ring",   "dip"),  ("ring",   "tip"),
    ("pinky",  "mcp"),  ("pinky",  "pip"),  ("pinky",  "dip"),  ("pinky",  "tip"),
]


def _make_lms(overrides: Optional[dict] = None) -> list[dict]:
    """Return 21 landmarks all at (0.5, 0.5, 0.0), with selective overrides by index."""
    lms = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "finger": f, "joint": j}
        for f, j in _LM_LABELS
    ]
    for idx, vals in (overrides or {}).items():
        lms[idx].update(vals)
    return lms


def _make_hand(label: str = "Right") -> HandLandmarks:
    return HandLandmarks(hand_label=label, landmarks=_make_lms())


# A real numpy frame that cv2 functions can operate on.
_FAKE_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
# A fake JPEG byte buffer (not a real JPEG, but has .tobytes() that returns bytes).
_FAKE_JPEG = np.zeros((300,), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Gesture classifier
# ---------------------------------------------------------------------------

class TestGestureClassifier(unittest.TestCase):

    # ── compute_pinch_distance ──────────────────────────────────────────────

    def test_pinch_distance_zero_when_tips_overlap(self):
        """Landmarks 4 and 8 at the same position → distance == 0."""
        lms = _make_lms()  # everything at (0.5, 0.5)
        self.assertAlmostEqual(compute_pinch_distance(lms), 0.0)

    def test_pinch_distance_correct(self):
        """Moving index tip 0.3 units to the right gives distance == 0.3."""
        lms = _make_lms({8: {"x": 0.8, "y": 0.5}})
        self.assertAlmostEqual(compute_pinch_distance(lms), 0.3, places=5)

    # ── compute_flexion_angles ──────────────────────────────────────────────

    def test_flexion_returns_five_values(self):
        self.assertEqual(len(compute_flexion_angles(_make_lms())), 5)

    def test_flexion_straight_finger_is_near_zero(self):
        """Index MCP→PIP→DIP collinear (vertical) → flexion ≈ 0°."""
        lms = _make_lms({
            5: {"x": 0.5, "y": 0.9},   # MCP
            6: {"x": 0.5, "y": 0.7},   # PIP
            7: {"x": 0.5, "y": 0.55},  # DIP
        })
        angles = compute_flexion_angles(lms)
        self.assertLess(angles[1], 5.0)  # index is element [1]

    def test_flexion_bent_finger_is_near_90(self):
        """PIP vectors perpendicular → flexion ≈ 90°."""
        lms = _make_lms({
            5: {"x": 0.5, "y": 0.7},  # MCP
            6: {"x": 0.5, "y": 0.6},  # PIP
            7: {"x": 0.6, "y": 0.6},  # DIP, 90° turn
        })
        angles = compute_flexion_angles(lms)
        self.assertAlmostEqual(angles[1], 90.0, delta=1.0)

    # ── classify_gesture ───────────────────────────────────────────────────

    _VALID_LABELS = {"open", "fist", "pinch", "point"}

    def test_classify_returns_valid_label(self):
        label, *_ = classify_gesture(_make_lms())
        self.assertIn(label, self._VALID_LABELS)

    def test_classify_confidence_in_unit_range(self):
        _, conf, *_ = classify_gesture(_make_lms())
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_classify_pinch_when_tips_overlap(self):
        """All landmarks at same point → thumb tip and index tip overlap → pinch."""
        label, _, dist, _ = classify_gesture(_make_lms())
        self.assertAlmostEqual(dist, 0.0)
        self.assertEqual(label, "pinch")

    def test_classify_open(self):
        """All fingers straight and spread → open."""
        # Build five straight fingers: each triplet collinear vertically, tips spread.
        overrides = {}
        for (a, b, c), x in zip(
            [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)],
            [0.25, 0.40, 0.50, 0.60, 0.70],
        ):
            overrides[a] = {"x": x, "y": 0.9}
            overrides[b] = {"x": x, "y": 0.7}
            overrides[c] = {"x": x, "y": 0.55}
        # Tips also spread (not in triplets but needed for pinch distance)
        overrides[4]  = {"x": 0.25, "y": 0.4}   # thumb tip
        overrides[8]  = {"x": 0.40, "y": 0.4}   # index tip
        overrides[12] = {"x": 0.50, "y": 0.4}
        overrides[16] = {"x": 0.60, "y": 0.4}
        overrides[20] = {"x": 0.70, "y": 0.4}
        label, *_ = classify_gesture(_make_lms(overrides))
        self.assertEqual(label, "open")

    def test_classify_fist(self):
        """All fingers bent 90° and thumb tip far from index tip → fist."""
        overrides = {4: {"x": 0.1, "y": 0.9}}  # thumb tip far away
        for mcp, pip_, dip_, tip, xb in [
            (5,  6,  7,  8,  0.40),
            (9,  10, 11, 12, 0.50),
            (13, 14, 15, 16, 0.60),
            (17, 18, 19, 20, 0.70),
        ]:
            overrides[mcp]  = {"x": xb,        "y": 0.8}
            overrides[pip_] = {"x": xb,        "y": 0.7}
            overrides[dip_] = {"x": xb + 0.1,  "y": 0.7}   # 90° bend at PIP
            overrides[tip]  = {"x": xb + 0.15, "y": 0.7}
        label, *_ = classify_gesture(_make_lms(overrides))
        self.assertEqual(label, "fist")

    def test_classify_point(self):
        """Index straight, middle/ring/pinky bent, thumb tip far away → point."""
        overrides = {
            # Straight index
            5:  {"x": 0.5, "y": 0.9}, 6:  {"x": 0.5, "y": 0.7},
            7:  {"x": 0.5, "y": 0.55}, 8: {"x": 0.5, "y": 0.4},
            # Middle bent 90°
            9:  {"x": 0.6, "y": 0.8}, 10: {"x": 0.6, "y": 0.7}, 11: {"x": 0.7, "y": 0.7},
            # Ring bent 90°
            13: {"x": 0.4, "y": 0.8}, 14: {"x": 0.4, "y": 0.7}, 15: {"x": 0.5, "y": 0.7},
            # Pinky bent 90°
            17: {"x": 0.3, "y": 0.8}, 18: {"x": 0.3, "y": 0.7}, 19: {"x": 0.4, "y": 0.7},
            # Thumb tip far from index tip
            4:  {"x": 0.1, "y": 0.9},
        }
        label, *_ = classify_gesture(_make_lms(overrides))
        self.assertEqual(label, "point")


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

class TestSceneBuilder(unittest.TestCase):

    def _result(self, hands=1, ts=1_000_000_000):
        return build_scene_update([_make_hand() for _ in range(hands)], timestamp_ns=ts)

    def test_empty_input_produces_no_entities(self):
        result = build_scene_update([], timestamp_ns=0)
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["deletions"], [])

    def test_one_hand_one_entity(self):
        self.assertEqual(len(self._result(1)["entities"]), 1)

    def test_two_hands_two_entities(self):
        self.assertEqual(len(self._result(2)["entities"]), 2)

    def test_entity_has_21_spheres(self):
        entity = self._result()["entities"][0]
        self.assertEqual(len(entity["spheres"]), 21)

    def test_entity_has_6_line_groups(self):
        # thumb / index / middle / ring / pinky / palm
        entity = self._result()["entities"][0]
        self.assertEqual(len(entity["lines"]), 6)

    def test_entity_lifetime_is_nonzero(self):
        entity = self._result()["entities"][0]
        self.assertGreater(
            entity["lifetime"]["sec"] * 1_000_000_000 + entity["lifetime"]["nsec"], 0
        )

    def test_entity_id_encodes_hand_label(self):
        hand = HandLandmarks(hand_label="Left", landmarks=_make_lms())
        entity = build_scene_update([hand], timestamp_ns=0)["entities"][0]
        self.assertIn("left", entity["id"])

    def test_sphere_colors_are_valid_rgba(self):
        entity = self._result()["entities"][0]
        for sphere in entity["spheres"]:
            color = sphere["color"]
            for ch in ("r", "g", "b", "a"):
                self.assertIn(ch, color)
                self.assertGreaterEqual(color[ch], 0.0)
                self.assertLessEqual(color[ch], 1.0)

    def test_timestamp_propagates_to_entity(self):
        ts = 9_876_543_210_000_000_000
        entity = build_scene_update([_make_hand()], timestamp_ns=ts)["entities"][0]
        self.assertEqual(entity["timestamp"]["sec"],  ts // 1_000_000_000)
        self.assertEqual(entity["timestamp"]["nsec"], ts %  1_000_000_000)

    def test_3d_positions_use_z_from_landmark(self):
        """Sphere z-coordinates must reflect the landmark z (negated + scaled)."""
        # Put landmark 0 (wrist) at a known z value.
        lms = _make_lms({0: {"x": 0.5, "y": 0.5, "z": 0.1}})
        hand = HandLandmarks(hand_label="Right", landmarks=lms)
        entity = build_scene_update([hand], timestamp_ns=0)["entities"][0]
        wrist_sphere = entity["spheres"][0]
        self.assertAlmostEqual(wrist_sphere["pose"]["position"]["z"], -0.1, places=5)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas(unittest.TestCase):
    """Schemas must be plain dicts (no $ref) and contain the expected top-level keys."""

    def _assert_no_refs(self, obj, path=""):
        """Recursively confirm no $ref key exists anywhere in the schema dict."""
        if isinstance(obj, dict):
            self.assertNotIn("$ref", obj, f"Unexpected $ref found at {path}")
            for k, v in obj.items():
                self._assert_no_refs(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._assert_no_refs(item, f"{path}[{i}]")

    def test_hand_landmarks_schema_required_fields(self):
        req = HAND_LANDMARKS_SCHEMA["required"]
        for field in ("timestamp", "hand_label", "landmarks"):
            self.assertIn(field, req)

    def test_hand_gesture_schema_required_fields(self):
        req = HAND_GESTURE_SCHEMA["required"]
        for field in ("gesture", "confidence", "pinch_distance", "flexion_angles"):
            self.assertIn(field, req)

    def test_scene_update_schema_required_fields(self):
        req = SCENE_UPDATE_SCHEMA["required"]
        for field in ("deletions", "entities"):
            self.assertIn(field, req)

    def test_compressed_image_schema_required_fields(self):
        req = COMPRESSED_IMAGE_SCHEMA["required"]
        for field in ("timestamp", "frame_id", "data", "format"):
            self.assertIn(field, req)

    def test_scene_update_schema_has_no_dollar_ref(self):
        """Schema was previously broken by $ref usage; confirm it is fully inlined."""
        self._assert_no_refs(SCENE_UPDATE_SCHEMA)

    def test_all_schemas_are_valid_json(self):
        """Every schema must round-trip through JSON without error."""
        for name, schema in [
            ("HandLandmarks",          HAND_LANDMARKS_SCHEMA),
            ("HandGesture",            HAND_GESTURE_SCHEMA),
            ("foxglove.SceneUpdate",   SCENE_UPDATE_SCHEMA),
            ("foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
        ]:
            with self.subTest(schema=name):
                serialised = json.dumps(schema)
                recovered  = json.loads(serialised)
                self.assertEqual(recovered["type"], "object")


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------

class TestServerIntegration(unittest.IsolatedAsyncioTestCase):
    """Start the app with mocked hardware and verify the Foxglove WebSocket layer."""

    async def test_server_starts_and_advertises_all_channels(self):
        import main as app

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _FAKE_FRAME)

        mock_tracker = MagicMock()
        mock_tracker.process.return_value = ([], [])
        mock_tracker.draw.return_value = _FAKE_FRAME

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=0),
            patch("cv2.destroyAllWindows"),
            patch("cv2.imencode", return_value=(True, _FAKE_JPEG)),
            patch.object(app, "HandTracker", return_value=mock_tracker),
        ):
            server_task = asyncio.create_task(app.main())

            # Give the server time to bind the port before connecting.
            await asyncio.sleep(0.4)

            self.assertFalse(
                server_task.done(),
                f"Server task exited prematurely: {server_task.exception() if server_task.done() else ''}",
            )

            try:
                async with websockets.connect(
                    "ws://localhost:8765",
                    subprotocols=["foxglove.websocket.v1"],
                    open_timeout=5.0,
                ) as ws:
                    # ── 1. serverInfo ──────────────────────────────────────
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    server_info = json.loads(raw)

                    self.assertEqual(server_info["op"], "serverInfo")
                    self.assertEqual(server_info["name"], "Hand Tracker")

                    # ── 2. advertise ───────────────────────────────────────
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    advertise = json.loads(raw)

                    self.assertEqual(advertise["op"], "advertise")

                    topics   = {ch["topic"]    for ch in advertise["channels"]}
                    schemas  = {ch["topic"]: ch for ch in advertise["channels"]}
                    encodings = {ch["encoding"] for ch in advertise["channels"]}

                    # All four topics present
                    for expected in (
                        "/hand/landmarks",
                        "/hand/gesture",
                        "/hand/markers",
                        "/camera/image",
                    ):
                        self.assertIn(expected, topics, f"Missing topic: {expected}")

                    # Every channel uses JSON encoding
                    self.assertEqual(encodings, {"json"}, "Expected all channels to use JSON encoding")

                    # Every channel schema is parseable JSON with a top-level "type" field
                    for topic, ch in schemas.items():
                        with self.subTest(topic=topic):
                            parsed = json.loads(ch["schema"])
                            self.assertIn(
                                "type", parsed,
                                f"Schema for {topic} is missing top-level 'type' field",
                            )

            finally:
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
