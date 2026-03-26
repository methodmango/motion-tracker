def _vec3():
    return {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}

def _color():
    return {"type": "object", "properties": {"r": {"type": "number"}, "g": {"type": "number"}, "b": {"type": "number"}, "a": {"type": "number"}}}

def _pose():
    return {
        "type": "object",
        "properties": {
            "position":    {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
            "orientation": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}},
        },
    }

def _time():
    return {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}

SCENE_UPDATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "foxglove.SceneUpdate",
    "type": "object",
    "properties": {
        "deletions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": _time(),
                    "type": {"type": "integer"},
                    "id":   {"type": "string"},
                },
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp":    _time(),
                    "frame_id":     {"type": "string"},
                    "id":           {"type": "string"},
                    "lifetime":     _time(),
                    "frame_locked": {"type": "boolean"},
                    "metadata":     {"type": "array", "items": {"type": "object"}},
                    "arrows":       {"type": "array", "items": {"type": "object"}},
                    "cubes":        {"type": "array", "items": {"type": "object"}},
                    "spheres": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pose":  _pose(),
                                "size":  _vec3(),
                                "color": _color(),
                            },
                        },
                    },
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type":            {"type": "integer"},
                                "pose":            _pose(),
                                "thickness":       {"type": "number"},
                                "scale_invariant": {"type": "boolean"},
                                "points":          {"type": "array", "items": _vec3()},
                                "color":           _color(),
                                "colors":          {"type": "array", "items": _color()},
                                "indices":         {"type": "array", "items": {"type": "integer"}},
                            },
                        },
                    },
                    "triangles": {"type": "array", "items": {"type": "object"}},
                    "texts":     {"type": "array", "items": {"type": "object"}},
                    "models":    {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
    "required": ["deletions", "entities"],
}

HAND_LANDMARKS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HandLandmarks",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "description": "Capture time",
            "properties": {
                "sec":  {"type": "integer"},
                "nsec": {"type": "integer"},
            },
            "required": ["sec", "nsec"],
        },
        "hand_label": {
            "type": "string",
            "description": "Left or Right",
        },
        "landmarks": {
            "type": "array",
            "description": "21 MediaPipe hand landmarks in normalized image space",
            "items": {
                "type": "object",
                "properties": {
                    "x":      {"type": "number", "description": "Normalized [0,1], left→right"},
                    "y":      {"type": "number", "description": "Normalized [0,1], top→bottom"},
                    "z":      {"type": "number", "description": "Depth relative to wrist (same scale as x)"},
                    "finger": {"type": "string", "description": "wrist|thumb|index|middle|ring|pinky"},
                    "joint":  {"type": "string", "description": "wrist|cmc|mcp|ip|pip|dip|tip"},
                },
                "required": ["x", "y", "z", "finger", "joint"],
            },
        },
    },
    "required": ["timestamp", "hand_label", "landmarks"],
}

COMPRESSED_IMAGE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "foxglove.CompressedImage",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec":  {"type": "integer"},
                "nsec": {"type": "integer"},
            },
            "required": ["sec", "nsec"],
        },
        "frame_id": {"type": "string"},
        "data":     {"type": "string", "contentEncoding": "base64"},
        "format":   {"type": "string", "description": "jpeg or png"},
    },
    "required": ["timestamp", "frame_id", "data", "format"],
}

PINCH_DISTANCE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PinchDistance",
    "type": "object",
    "properties": {
        "timestamp":  _time(),
        "hand_label": {"type": "string"},
        "value":      {"type": "number", "description": "Normalized thumb-tip to index-tip distance"},
    },
    "required": ["timestamp", "hand_label", "value"],
}

FINGER_ANGLES_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FingerAngles",
    "type": "object",
    "properties": {
        "timestamp":  _time(),
        "hand_label": {"type": "string"},
        "thumb":      {"type": "number", "description": "Flexion angle in degrees"},
        "index":      {"type": "number", "description": "Flexion angle in degrees"},
        "middle":     {"type": "number", "description": "Flexion angle in degrees"},
        "ring":       {"type": "number", "description": "Flexion angle in degrees"},
        "pinky":      {"type": "number", "description": "Flexion angle in degrees"},
    },
    "required": ["timestamp", "hand_label", "thumb", "index", "middle", "ring", "pinky"],
}

HAND_VELOCITY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HandVelocity",
    "type": "object",
    "properties": {
        "timestamp":  _time(),
        "hand_label": {"type": "string"},
        "vx":         {"type": "number", "description": "Wrist x-velocity (normalized units/sec)"},
        "vy":         {"type": "number", "description": "Wrist y-velocity (normalized units/sec)"},
        "speed":      {"type": "number", "description": "Wrist speed magnitude (normalized units/sec)"},
    },
    "required": ["timestamp", "hand_label", "vx", "vy", "speed"],
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
            "description": "Rule-based gesture label: open, fist, pinch, or point",
        },
        "confidence": {
            "type": "number",
            "description": "Gesture confidence score [0, 1]",
        },
        "pinch_distance": {
            "type": "number",
            "description": "Normalized distance between thumb tip (4) and index tip (8)",
        },
        "flexion_angles": {
            "type": "array",
            "description": "Per-finger flexion in degrees [thumb, index, middle, ring, pinky]; 0=straight",
            "items": {"type": "number"},
            "minItems": 5,
            "maxItems": 5,
        },
    },
    "required": ["hand_label", "gesture", "confidence", "pinch_distance", "flexion_angles"],
}
