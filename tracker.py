import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from gesture_classifier import classify_gesture

# Model bundle — downloaded on first run
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
_MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

# Semantic label for each of the 21 MediaPipe hand landmarks.
# Index in the list == landmark index returned by the model.
_LM_LABELS: list[tuple[str, str]] = [
    ("wrist",  "wrist"),
    ("thumb",  "cmc"), ("thumb",  "mcp"), ("thumb",  "ip"),  ("thumb",  "tip"),
    ("index",  "mcp"), ("index",  "pip"), ("index",  "dip"), ("index",  "tip"),
    ("middle", "mcp"), ("middle", "pip"), ("middle", "dip"), ("middle", "tip"),
    ("ring",   "mcp"), ("ring",   "pip"), ("ring",   "dip"), ("ring",   "tip"),
    ("pinky",  "mcp"), ("pinky",  "pip"), ("pinky",  "dip"), ("pinky",  "tip"),
]

# Standard MediaPipe hand skeleton — 21 landmarks, fixed by the model spec
_HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
])


def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        print(f"Downloading gesture recognizer model → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return str(_MODEL_PATH)


@dataclass
class HandLandmarks:
    hand_label: str
    landmarks: list[dict]  # [{"x", "y", "z", "finger", "joint"}, ...]  len==21


@dataclass
class HandGesture:
    hand_label: str
    gesture: str
    confidence: float
    pinch_distance: float = 0.0
    flexion_angles: list[float] = field(default_factory=lambda: [0.0] * 5)


class HandTracker:
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7):
        model_path = _ensure_model()

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def process(self, frame) -> tuple[list[HandLandmarks], list[HandGesture]]:
        """Process a BGR frame. Returns (landmarks_list, gestures_list)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        landmarks_list: list[HandLandmarks] = []
        gestures_list: list[HandGesture] = []

        if not result.hand_landmarks:
            return landmarks_list, gestures_list

        for i, hand_lms in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].display_name  # "Left" or "Right"

            lm_list = [
                {
                    "x": lm.x, "y": lm.y, "z": lm.z,
                    "finger": _LM_LABELS[j][0],
                    "joint":  _LM_LABELS[j][1],
                }
                for j, lm in enumerate(hand_lms)
            ]
            landmarks_list.append(HandLandmarks(hand_label=label, landmarks=lm_list))

            gesture, confidence, pinch_dist, flexions = classify_gesture(lm_list)
            gestures_list.append(
                HandGesture(
                    hand_label=label,
                    gesture=gesture,
                    confidence=confidence,
                    pinch_distance=pinch_dist,
                    flexion_angles=flexions,
                )
            )

        return landmarks_list, gestures_list

    def draw(self, frame, landmarks_list: list[HandLandmarks]):
        """Return a copy of frame with hand landmarks drawn."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for hand_lms in landmarks_list:
            for start_idx, end_idx in _HAND_CONNECTIONS:
                a = hand_lms.landmarks[start_idx]
                b = hand_lms.landmarks[end_idx]
                cv2.line(
                    annotated,
                    (int(a["x"] * w), int(a["y"] * h)),
                    (int(b["x"] * w), int(b["y"] * h)),
                    (0, 200, 0),
                    1,
                )
            for lm in hand_lms.landmarks:
                cv2.circle(annotated, (int(lm["x"] * w), int(lm["y"] * h)), 4, (0, 0, 255), -1)

            wrist = hand_lms.landmarks[0]
            cv2.putText(
                annotated,
                hand_lms.hand_label,
                (int(wrist["x"] * w), int(wrist["y"] * h) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return annotated

    def close(self) -> None:
        self._recognizer.close()
