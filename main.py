"""
Hand tracking application with Foxglove WebSocket logging.

Connect Foxglove Studio to ws://localhost:8765 to visualize and record data.
Topics:
  /hand/landmarks  — HandLandmarks (21 normalized 2D points per hand)
  /hand/gesture    — HandGesture   (detected gesture + confidence)

Press 'q' in the preview window to quit.
"""

import asyncio
import json
import time

import cv2
from foxglove_websocket.server import FoxgloveServer

from schemas import HAND_GESTURE_SCHEMA, HAND_LANDMARKS_SCHEMA
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

                for hand_lms in landmarks_list:
                    payload = json.dumps(
                        {
                            "hand_label": hand_lms.hand_label,
                            "landmarks": hand_lms.landmarks,
                        }
                    ).encode()
                    await server.send_message(landmarks_chan, timestamp_ns, payload)

                for gesture in gestures_list:
                    payload = json.dumps(
                        {
                            "hand_label": gesture.hand_label,
                            "gesture": gesture.gesture,
                            "confidence": gesture.confidence,
                        }
                    ).encode()
                    await server.send_message(gesture_chan, timestamp_ns, payload)

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
