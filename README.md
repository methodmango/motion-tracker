# motion-tracker

Real-time hand tracking with [MediaPipe](https://developers.google.com/mediapipe) streamed to [Foxglove Studio](https://foxglove.dev) over WebSocket.

## Requirements

- Python 3.9+
- A webcam

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The MediaPipe gesture recognizer model (~25 MB) is downloaded automatically on first run.

## Running

```bash
python main.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device N` | `0` | Webcam device index (`/dev/video0`, `0` on macOS) |
| `--port N` | `8765` | WebSocket port Foxglove Studio connects to |

Examples:

```bash
python main.py --device 1          # use a second camera
python main.py --port 8888         # non-default port
python main.py --device 0 --port 8765
```

Press **`q`** in the preview window or **Ctrl+C** in the terminal to quit cleanly.

## Connecting Foxglove Studio

1. Open [Foxglove Studio](https://app.foxglove.dev) (desktop or web).
2. Click **Open connection → Foxglove WebSocket**.
3. Enter `ws://localhost:8765` (adjust port if you used `--port`).
4. Click **Open**.

## Topics

| Topic | Schema | Contents |
|-------|--------|----------|
| `/hand/landmarks` | `HandLandmarks` | 21 MediaPipe landmarks (x, y, z, finger, joint) |
| `/hand/gesture` | `HandGesture` | Label, confidence, pinch distance, flexion angles array |
| `/hand/markers` | `foxglove.SceneUpdate` | 3D spheres + skeleton lines, colored per finger |
| `/hand/pinch_distance` | `PinchDistance` | Normalized thumb-to-index distance (`value` field) |
| `/hand/finger_angles` | `FingerAngles` | Per-finger flexion in degrees (`thumb`, `index`, `middle`, `ring`, `pinky`) |
| `/hand/velocity` | `HandVelocity` | Wrist velocity (`vx`, `vy`, `speed` in normalized units/sec) |
| `/camera/image` | `foxglove.CompressedImage` | JPEG webcam frame |

## Suggested Foxglove Studio Layout

### 4-panel setup

**Panel 1 — 3D view (top-left)**
- Add a **3D** panel.
- It will auto-detect `/hand/markers` and render the skeleton.
- Set **Follow mode** to `Fixed frame`, frame ID `hand`.

**Panel 2 — Camera feed (top-right)**
- Add an **Image** panel.
- Set topic to `/camera/image`.

**Panel 3 — Plot: pinch + velocity (bottom-left)**
- Add a **Plot** panel.
- Add series:
  - `/hand/pinch_distance.value` — label `pinch`
  - `/hand/velocity.speed` — label `speed`
  - `/hand/velocity.vx` — label `vx` (optional)
  - `/hand/velocity.vy` — label `vy` (optional)
- Set Y-axis range `0 – 1.5` as a starting point.

**Panel 4 — Plot: finger angles (bottom-right)**
- Add a **Plot** panel.
- Add series:
  - `/hand/finger_angles.thumb` — label `thumb`
  - `/hand/finger_angles.index` — label `index`
  - `/hand/finger_angles.middle` — label `middle`
  - `/hand/finger_angles.ring` — label `ring`
  - `/hand/finger_angles.pinky` — label `pinky`
- Set Y-axis range `0 – 180` (degrees).

**Optional — Raw messages**
- Add a **Raw Messages** panel and point it to `/hand/gesture` to see the live gesture label and confidence.

### Tips

- Use the **Record** button (top bar) to capture a `.mcap` file for offline replay.
- In the Plot panels, enable **Sync** (⚙ icon) so both plots scroll together.
- Right-click a series in the legend to change its color.
