from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from .alerts import console_alert
from .detector import DetectorConfig, ShortCircuitDetector


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


class _FfmpegCapture:
    """Minimal VideoCapture-compatible reader for codecs OpenCV cannot decode."""

    def __init__(self, path: str) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise RuntimeError(
                "OpenCV could not decode the video and ffmpeg/ffprobe are not installed"
            )
        metadata = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "default=noprint_wrappers=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        values = dict(line.split("=", 1) for line in metadata if "=" in line)
        try:
            self.width = int(values["width"])
            self.height = int(values["height"])
            numerator, denominator = values["r_frame_rate"].split("/", 1)
            self.fps = float(numerator) / float(denominator)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(f"unable to read video metadata: {path}") from exc
        self._frame_bytes = self.width * self.height * 3
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self._process.poll() is None

    def read(self) -> tuple[bool, np.ndarray]:
        if self._process.stdout is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        data = self._read_frame()
        if data is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return True, frame

    def _read_frame(self) -> bytes | None:
        if self._process.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = self._frame_bytes
        while remaining:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect short-circuit/arc flashes from an RGB camera or video."
    )
    parser.add_argument("--source", default="0", help="camera index or video path (default: 0)")
    parser.add_argument("--output", type=Path, help="optional annotated video path")
    parser.add_argument("--no-display", action="store_true", help="run without an OpenCV window")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="keep a camera process alive and reconnect after camera failures",
    )
    parser.add_argument(
        "--reconnect-seconds",
        type=float,
        default=5.0,
        help="seconds between camera reconnect attempts (default: 5)",
    )
    parser.add_argument(
        "--bright-value",
        type=int,
        default=200,
        help="minimum HSV value for bright arc pixels (default: 200)",
    )
    parser.add_argument(
        "--min-local-contrast",
        type=int,
        default=18,
        help="minimum value above the local Gaussian background (default: 18)",
    )
    parser.add_argument(
        "--min-strong-rise",
        type=int,
        default=220,
        help="single-component rise threshold for tiny sparks (default: 220)",
    )
    parser.add_argument(
        "--min-cluster-area",
        type=int,
        default=100,
        help="minimum combined area for a bright spatial arc cluster (default: 100)",
    )
    parser.add_argument(
        "--min-cluster-brightness",
        type=float,
        default=235.0,
        help="minimum weighted HSV brightness for an arc cluster (default: 235)",
    )
    parser.add_argument(
        "--event-frames",
        type=int,
        default=1,
        help="consecutive rising frames required to trigger (default: 1)",
    )
    parser.add_argument("--process-width", type=int, default=640)
    parser.add_argument(
        "--roi",
        help="monitored zone as x,y,width,height; strongly recommended for substation use",
    )
    parser.add_argument("--cooldown-frames", type=int, default=15)
    return parser


def run(
    source: int | str,
    *,
    output: Path | None = None,
    display: bool = True,
    config: DetectorConfig | None = None,
    alert_stream: TextIO | None = None,
    continuous: bool = False,
    reconnect_seconds: float = 5.0,
) -> int:
    if reconnect_seconds <= 0:
        raise ValueError("reconnect_seconds must be greater than zero")
    detector = ShortCircuitDetector(config)
    writer: cv2.VideoWriter | None = None
    capture: cv2.VideoCapture | _FfmpegCapture | None = None
    frames_read = 0
    ffmpeg_fallback_attempted = False
    try:
        while True:
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    if not continuous:
                        raise RuntimeError(f"unable to open video source: {source}")
                    print(
                        f"Camera unavailable; retrying in {reconnect_seconds:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(reconnect_seconds)
                    continue
                detector.reset()
            ok, frame = capture.read()
            if not ok:
                if (
                    frames_read == 0
                    and not ffmpeg_fallback_attempted
                    and isinstance(source, str)
                    and Path(source).is_file()
                ):
                    capture.release()
                    print(
                        "OpenCV could not decode the video; retrying through ffmpeg",
                        file=sys.stderr,
                        flush=True,
                    )
                    capture = _FfmpegCapture(source)
                    ffmpeg_fallback_attempted = True
                    detector.reset()
                    continue
                capture.release()
                capture = None
                if continuous:
                    print(
                        f"Camera stream lost; retrying in {reconnect_seconds:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(reconnect_seconds)
                    continue
                if frames_read == 0:
                    raise RuntimeError(f"video source produced no decodable frames: {source}")
                break
            frames_read += 1
            detection = detector.process(frame)
            if detector.should_alert(detection):
                if alert_stream is None:
                    console_alert(detection)
                else:
                    print(
                        f"ALERT: {detection.message} confidence={detection.confidence:.0%}",
                        file=alert_stream,
                        flush=True,
                    )
            annotated = frame.copy()
            for x, y, width, height in detection.boxes:
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                detection.message,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if detection.detected else (0, 200, 0),
                2,
            )
            if output is not None:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
                    size = (annotated.shape[1], annotated.shape[0])
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"unable to create output video: {output}")
                writer.write(annotated)
            if display:
                cv2.imshow("Short-circuit detector (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return 0


def _roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return (parts[0], parts[1], parts[2], parts[3])


def main() -> int:
    args = build_parser().parse_args()
    config = DetectorConfig(
        bright_value=args.bright_value,
        min_local_contrast=args.min_local_contrast,
        min_strong_rise=args.min_strong_rise,
        min_cluster_area=args.min_cluster_area,
        min_cluster_brightness=args.min_cluster_brightness,
        event_frames=args.event_frames,
        cooldown_frames=args.cooldown_frames,
        process_width=args.process_width,
        roi=_roi(args.roi),
    )
    source = _source(args.source)
    return run(
        source,
        output=args.output,
        display=not args.no_display,
        config=config,
        continuous=args.continuous or isinstance(source, int),
        reconnect_seconds=args.reconnect_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
