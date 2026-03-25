"""
Hand tracking application with Foxglove WebSocket logging.

Connect Foxglove Studio to ws://localhost:8765 to visualize and record data.
Topics:
  /hand/landmarks  — HandLandmarks        (21 landmarks with x,y,z + finger/joint labels)
  /hand/gesture    — HandGesture          (rule-based label, confidence, pinch distance, flexion angles)
  /hand/markers    — foxglove.SceneUpdate (3D spheres + skeleton lines, colored per finger)
  /camera/image    — foxglove.CompressedImage (JPEG-compressed webcam frame)

Press 'q' in the preview window to quit.
"""

import asyncio
import base64
import json
import time

import cv2
from foxglove_websocket.server import FoxgloveServer

from scene_builder import build_scene_update
from schemas import COMPRESSED_IMAGE_SCHEMA, HAND_GESTURE_SCHEMA, HAND_LANDMARKS_SCHEMA, SCENE_UPDATE_SCHEMA
from tracker import HandTracker


async def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tracker = HandTracker()

    async with FoxgloveServer("0.0.0.0", 8765, "Hand Tracker") as server:
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

        print("Foxglove WebSocket server: ws://localhost:8765")
        print("Open Foxglove Studio and connect to that address.")
        print("Press 'q' in the preview window to quit.\n")

        try:
            while True:
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

                scene = build_scene_update(landmarks_list, timestamp_ns)
                await server.send_message(
                    markers_chan, timestamp_ns, json.dumps(scene).encode()
                )

                annotated = tracker.draw(frame, landmarks_list)
                cv2.imshow("Hand Tracker (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Yield to the event loop so the WebSocket can flush messages
                await asyncio.sleep(0)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            tracker.close()


if __name__ == "__main__":
    asyncio.run(main())
