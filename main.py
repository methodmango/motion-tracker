"""
Hand tracking application with Foxglove WebSocket logging.

Connect Foxglove Studio to ws://localhost:8765 to visualize and record data.
Topics:
  /hand/landmarks      — HandLandmarks        (21 landmarks with x,y,z + finger/joint labels)
  /hand/gesture        — HandGesture          (rule-based label, confidence, pinch distance, flexion angles)
  /hand/markers        — foxglove.SceneUpdate (3D spheres + skeleton lines, colored per finger)
  /hand/pinch_distance — PinchDistance        (normalized thumb-to-index distance, plottable)
  /hand/finger_angles  — FingerAngles         (per-finger flexion in degrees, plottable)
  /hand/velocity       — HandVelocity         (wrist vx/vy/speed in normalized units/sec, plottable)
  /camera/image        — foxglove.CompressedImage (JPEG-compressed webcam frame)

Press 'q' in the preview window or Ctrl+C to quit.
"""

import argparse
import asyncio
import base64
import json
import math
import signal
import time

import cv2
from foxglove_websocket.server import FoxgloveServer

from scene_builder import build_scene_update
from schemas import (
    COMPRESSED_IMAGE_SCHEMA,
    FINGER_ANGLES_SCHEMA,
    HAND_GESTURE_SCHEMA,
    HAND_LANDMARKS_SCHEMA,
    HAND_VELOCITY_SCHEMA,
    PINCH_DISTANCE_SCHEMA,
    SCENE_UPDATE_SCHEMA,
)
from tracker import HandTracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand tracking with Foxglove WebSocket logging."
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket port for Foxglove Studio (default: 8765)",
    )
    return parser.parse_args()


async def main(device: int, port: int) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")

    tracker = HandTracker()
    # Previous wrist position per hand label for velocity: {label: (x, y, timestamp_ns)}
    _prev_wrist: dict[str, tuple[float, float, int]] = {}

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with FoxgloveServer("0.0.0.0", port, "Hand Tracker") as server:
        landmarks_chan = await server.add_channel(
            {
                "topic": "/hand/landmarks",
                "encoding": "json",
                "schemaName": "HandLandmarks",
                "schema": json.dumps(HAND_LANDMARKS_SCHEMA),
            }
        )
        gesture_chan = await server.add_channel(
            {
                "topic": "/hand/gesture",
                "encoding": "json",
                "schemaName": "HandGesture",
                "schema": json.dumps(HAND_GESTURE_SCHEMA),
            }
        )
        markers_chan = await server.add_channel(
            {
                "topic": "/hand/markers",
                "encoding": "json",
                "schemaName": "foxglove.SceneUpdate",
                "schema": json.dumps(SCENE_UPDATE_SCHEMA),
            }
        )
        image_chan = await server.add_channel(
            {
                "topic": "/camera/image",
                "encoding": "json",
                "schemaName": "foxglove.CompressedImage",
                "schema": json.dumps(COMPRESSED_IMAGE_SCHEMA),
            }
        )
        pinch_chan = await server.add_channel(
            {
                "topic": "/hand/pinch_distance",
                "encoding": "json",
                "schemaName": "PinchDistance",
                "schema": json.dumps(PINCH_DISTANCE_SCHEMA),
            }
        )
        finger_angles_chan = await server.add_channel(
            {
                "topic": "/hand/finger_angles",
                "encoding": "json",
                "schemaName": "FingerAngles",
                "schema": json.dumps(FINGER_ANGLES_SCHEMA),
            }
        )
        velocity_chan = await server.add_channel(
            {
                "topic": "/hand/velocity",
                "encoding": "json",
                "schemaName": "HandVelocity",
                "schema": json.dumps(HAND_VELOCITY_SCHEMA),
            }
        )

        print(f"Foxglove WebSocket server: ws://localhost:{port}")
        print("Open Foxglove Studio and connect to that address.")
        print("Press 'q' in the preview window or Ctrl+C to quit.\n")

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame — stopping.")
                    break

                landmarks_list, gestures_list = tracker.process(frame)
                timestamp_ns = time.time_ns()

                ts = {"sec": timestamp_ns // 1_000_000_000, "nsec": timestamp_ns % 1_000_000_000}

                for hand_lms in landmarks_list:
                    payload = json.dumps(
                        {
                            "timestamp":  ts,
                            "hand_label": hand_lms.hand_label,
                            "landmarks":  hand_lms.landmarks,
                        }
                    ).encode()
                    await server.send_message(landmarks_chan, timestamp_ns, payload)

                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_payload = json.dumps(
                    {
                        "timestamp": ts,
                        "frame_id":  "camera",
                        "data":      base64.b64encode(jpeg_buf.tobytes()).decode(),
                        "format":    "jpeg",
                    }
                ).encode()
                await server.send_message(image_chan, timestamp_ns, image_payload)

                for gesture in gestures_list:
                    payload = json.dumps(
                        {
                            "hand_label": gesture.hand_label,
                            "gesture": gesture.gesture,
                            "confidence": gesture.confidence,
                            "pinch_distance": gesture.pinch_distance,
                            "flexion_angles": gesture.flexion_angles,
                        }
                    ).encode()
                    await server.send_message(gesture_chan, timestamp_ns, payload)

                    await server.send_message(
                        pinch_chan,
                        timestamp_ns,
                        json.dumps({"timestamp": ts, "hand_label": gesture.hand_label, "value": gesture.pinch_distance}).encode(),
                    )

                    angles = gesture.flexion_angles  # [thumb, index, middle, ring, pinky]
                    await server.send_message(
                        finger_angles_chan,
                        timestamp_ns,
                        json.dumps({
                            "timestamp":  ts,
                            "hand_label": gesture.hand_label,
                            "thumb":      angles[0],
                            "index":      angles[1],
                            "middle":     angles[2],
                            "ring":       angles[3],
                            "pinky":      angles[4],
                        }).encode(),
                    )

                for hand_lms in landmarks_list:
                    wrist = hand_lms.landmarks[0]
                    wx, wy = wrist["x"], wrist["y"]
                    label = hand_lms.hand_label
                    prev = _prev_wrist.get(label)
                    if prev is not None:
                        px, py, pt_ns = prev
                        dt = (timestamp_ns - pt_ns) * 1e-9
                        if dt > 0:
                            vx = (wx - px) / dt
                            vy = (wy - py) / dt
                            speed = math.hypot(vx, vy)
                            await server.send_message(
                                velocity_chan,
                                timestamp_ns,
                                json.dumps({"timestamp": ts, "hand_label": label, "vx": vx, "vy": vy, "speed": speed}).encode(),
                            )
                    _prev_wrist[label] = (wx, wy, timestamp_ns)

                scene = build_scene_update(landmarks_list, timestamp_ns)
                await server.send_message(
                    markers_chan, timestamp_ns, json.dumps(scene).encode()
                )

                annotated = tracker.draw(frame, landmarks_list)
                cv2.imshow("Hand Tracker (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()

                # Yield to the event loop so the WebSocket can flush messages
                await asyncio.sleep(0)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            tracker.close()
            print("Shutting down.")


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.device, args.port))
