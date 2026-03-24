HAND_LANDMARKS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HandLandmarks",
    "type": "object",
    "properties": {
        "hand_label": {
            "type": "string",
            "description": "Left or Right",
        },
        "landmarks": {
            "type": "array",
            "description": "21 MediaPipe hand landmarks, normalized [0,1]",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["id", "x", "y"],
            },
        },
    },
    "required": ["hand_label", "landmarks"],
}

HAND_GESTURE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HandGesture",
    "type": "object",
    "properties": {
        "hand_label": {
            "type": "string",
            "description": "Left or Right",
        },
        "gesture": {
            "type": "string",
            "description": "Detected gesture name",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score [0,1]",
        },
    },
    "required": ["hand_label", "gesture", "confidence"],
}
