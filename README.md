# RGB Short-Circuit Flash Detector

This is a real-time RGB detector based on the supplied paper,
*Advancement in Electrical Grid Fault Reporting using Artificial Intelligence
and Computer Vision*. It detects localized arc-like flashes rather than
classifying every bright object as a fault. The detector combines HSV
brightness, local contrast, positive frame-to-frame rise, connected components,
and spatial clustering. This is important for pole and wire footage where the
sky can occupy most of the image.

## Requirements

- Python 3.10 or newer
- OpenCV and NumPy (installed from `requirements.txt`)
- `ffmpeg` and `ffprobe` are optional, but required as a fallback for video
  codecs that OpenCV cannot decode

## Install

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Camera

Start the detector using the default camera (`0`):

```bash
python -m short_detector.cli --source 0
```

Press `q` in the display window to stop. An alert is printed and the terminal
bell is emitted once per detected event.

### Video file

Process a local video without opening a display window and save an annotated
copy:

```bash
python -m short_detector.cli \
  --source videos/input.webm \
  --no-display \
  --output videos/results/input_annotated.mp4
```

If OpenCV cannot decode a local video (for example, AV1 WebM), the CLI
automatically retries through `ffmpeg` when `ffmpeg` and `ffprobe` are
installed.

### Tests

Run the test suite from the project directory:

```bash
PYTHONPATH=. pytest -q
```

The default detector accepts a one-frame rise so a short capacitor spark is
not missed, then waits for the falling edge (or a bounded sustained flash)
before emitting one event. A cooldown prevents alert floods.

### Camera setup

For reliable results, lock exposure, white balance, and focus as recommended
by the paper. Point the camera at the suspected connection and keep the
camera fixed. RGB alone cannot reliably detect an electrical fault that has
no visible flash; thermal/UV sensing and a trained object detector should be
added for those cases.

### Unattended continuous operation

Camera index `0` automatically runs in continuous mode and reconnects after a
USB/network camera interruption:

```bash
python -m short_detector.cli \
  --source 0 \
  --no-display \
  --continuous \
  --roi 100,200,500,400 \
  --reconnect-seconds 5 \
  --output videos/results/live_annotated.mp4
```

For an RTSP stream or another camera URL, explicitly enable continuous mode:

```bash
python -m short_detector.cli \
  --source rtsp://USER:PASSWORD@CAMERA_IP/stream \
  --continuous \
  --no-display
```

Stop the service with `Ctrl+C`. The program emits one alert per event rather
than repeating alerts while the same event remains active. Run it under
systemd, Docker, or a process supervisor if it must restart after a machine
reboot. Protect RTSP credentials and do not place them in shared shell history.

## Detection behavior

Each frame is converted to HSV. A candidate pixel must have value at least
200, sit at least 18 levels above its local Gaussian background, and rise by
at least 25 levels from the previous frame. The detector keeps small
components, combines nearby components into an arc cluster, and requires the
cluster to be bright and spatially compact. A very strong single-component
rise is allowed for tiny sparks. Broad scene changes are rejected by area and
global-change limits. Static bright insulators, clouds, and sky glare therefore
do not trigger simply because they are bright.

For a substation camera, configure one ROI per monitored joint/conductor
instead of analyzing the entire image. The ROI format is
`x,y,width,height`:

```bash
python -m short_detector.cli --source 0 --continuous --no-display \
  --roi 100,200,500,400
```

The camera input is automatically reduced to a maximum processing width of
640 pixels, which is appropriate for a 720p laptop camera and keeps CPU use
low. Original-resolution ROI coordinates and detection boxes are scaled
automatically. For difficult scenes, tune the detector without changing code:

```bash
python -m short_detector.cli --source 0 --continuous --no-display \
  --roi 100,200,500,400 \
  --bright-value 200 \
  --min-local-contrast 18 \
  --min-strong-rise 220
```

Use `python -m short_detector.cli --help` to see all available command-line
options, including thresholds, cooldown, processing width, and reconnect
settings.

An ROI around the pole, conductor, or capacitor is strongly recommended. It
is the most reliable way to prevent unrelated sky motion from being considered
at all; the detector cannot infer electrical relevance from RGB pixels alone.

## Important model limitation

The supplied paper describes value clamping and recommends a YOLO/FOMO model,
but it does not provide a labeled training dataset or trained weights. It is
therefore not possible to honestly train a business-grade spark classifier
from the paper and the single unlabeled video alone. The pulse gate above is a
safer interim detector, not proof of an electrical fault. For production
deployment, collect labeled clips from the actual camera view (`spark`,
`normal_white`, `sun_glare`, `headlights`, `rain`, `insects`, and `camera
exposure_change`) and train/validate a detector with a false-alarm target
before connecting it to protection equipment. Keep hardware protection relays
independent of this software.

This output is an inspection aid, not a replacement for electrical protection
relays or a qualified safety procedure. RGB cannot detect a spark that is not
visible in the image; use UV, thermal, current, or other sensors when the
electrical event is invisible or heavily occluded.
