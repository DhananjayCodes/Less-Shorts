from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds for RGB arc/short-flash detection.

    An arc is treated as a small, high-contrast transient.  Brightness alone
    is deliberately not enough: clouds and sky glare can be brighter than an
    arc, but are usually diffuse and do not produce a fast local rise.
    """

    bright_value: int = 200
    max_brightness_saturation: int = 255
    min_color_delta: int = 25
    min_local_contrast: int = 18
    min_area: int = 2
    min_bright_fraction: float = 0.00015
    min_changed_fraction: float = 0.0001
    event_frames: int = 1
    max_flash_frames: int = 6
    min_flash_area: int = 2
    min_flash_brightness: int = 220
    max_flash_fraction: float = 0.04
    max_flash_components: int = 100
    cluster_radius: float = 90.0
    min_cluster_area: int = 100
    min_cluster_brightness: float = 235.0
    min_strong_rise: int = 220
    min_flash_compactness: float = 0.02
    max_global_change_fraction: float = 0.25
    cooldown_frames: int = 15
    process_width: int = 640
    roi: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class Detection:
    detected: bool
    confidence: float
    changed_fraction: float
    bright_fraction: float
    boxes: tuple[tuple[int, int, int, int], ...]
    message: str


@dataclass(frozen=True)
class _FlashComponent:
    box: tuple[int, int, int, int]
    area: int
    mean_value: float
    max_rise: int
    center: tuple[float, float]


class ShortCircuitDetector:
    """Detect short-circuit-like bright flashes in an RGB/BGR video stream."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._previous: Optional[np.ndarray] = None
        self._rise_frame: Optional[int] = None
        self._rise_mask: Optional[np.ndarray] = None
        self._rise_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self._candidate_streak = 0
        self._frame_index = 0
        self._cooldown = 0
        self._active = False
        self._alerted = False
        self._processing_roi: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self._previous = None
        self._rise_frame = None
        self._rise_mask = None
        self._rise_boxes = ()
        self._candidate_streak = 0
        self._frame_index = 0
        self._cooldown = 0
        self._active = False
        self._alerted = False
        self._processing_roi = None

    def process(self, frame: np.ndarray) -> Detection:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (height, width, 3)")
        if frame.dtype != np.uint8:
            raise ValueError("frame must have dtype uint8")
        self._frame_index += 1
        original_height, original_width = frame.shape[:2]
        scale = min(1.0, self.config.process_width / max(original_width, 1))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        if self.config.roi is None:
            self._processing_roi = None
        else:
            x, y, width, height = self.config.roi
            self._processing_roi = (
                round(x * scale),
                round(y * scale),
                round(width * scale),
                round(height * scale),
            )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        bright = (value >= self.config.bright_value) & (
            saturation <= self.config.max_brightness_saturation
        )

        if self._previous is None:
            self._previous = frame.copy()
            return self._result(False, 0.0, 0.0, float(bright.mean()), ())

        previous = self._previous
        previous_value = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)[:, :, 2]
        delta = np.max(cv2.absdiff(frame, previous), axis=2)
        local_background = cv2.GaussianBlur(value, (0, 0), 7)
        local_contrast = value.astype(np.int16) - local_background.astype(np.int16)
        value_rise = value.astype(np.int16) - previous_value.astype(np.int16)
        changed = (
            bright
            & (local_contrast >= self.config.min_local_contrast)
            & (value_rise >= self.config.min_color_delta)
            & self._roi_mask(frame.shape[:2])
        )

        bright_fraction = float(bright.mean())
        changed_fraction = float(changed.mean())
        global_change_fraction = float(
            (delta >= self.config.min_color_delta).mean()
        )
        candidate = changed.astype(np.uint8) * 255
        # Close only one-pixel gaps.  An opening here used to erase the small
        # capacitor sparks this detector is intended to catch.
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
        )
        component_min_area = max(
            1, min(self.config.min_area, self.config.min_flash_area)
        )
        components = self._components(
            candidate,
            value,
            value_rise,
            component_min_area,
            self.config.min_flash_brightness,
        )
        components = self._select_components(components)
        rise_boxes = tuple(component.box for component in components)
        candidate = self._component_mask(candidate, components)
        candidate_fraction = float((candidate > 0).mean())
        rise = (
            bright_fraction >= self.config.min_bright_fraction
            and changed_fraction >= self.config.min_changed_fraction
            and candidate_fraction <= self.config.max_flash_fraction
            and len(components) <= self.config.max_flash_components
            and global_change_fraction <= self.config.max_global_change_fraction
            and bool(rise_boxes)
        )
        pulse = False
        boxes = rise_boxes
        if rise:
            self._candidate_streak += 1
        else:
            self._candidate_streak = 0
        event_ready = rise and self._candidate_streak >= max(1, self.config.event_frames)
        if event_ready and self._rise_frame is None:
            self._rise_frame = self._frame_index
            self._rise_mask = candidate > 0
            self._rise_boxes = rise_boxes
        elif self._rise_frame is not None:
            elapsed = self._frame_index - self._rise_frame
            if elapsed > self.config.max_flash_frames:
                # A flash can be sustained or flickering.  Do not wait
                # indefinitely for a clean falling edge.
                pulse = True
                boxes = self._rise_boxes
                self._rise_frame = None
                self._rise_mask = None
                self._rise_boxes = ()
            elif self._rise_mask is not None and not rise:
                # A real visible flash normally has a falling edge.  The
                # candidate itself already excludes diffuse sky changes.
                rise_area = int(self._rise_mask.sum())
                frame_area = self._rise_mask.shape[0] * self._rise_mask.shape[1]
                pulse = bool(
                    elapsed <= self.config.max_flash_frames
                    and rise_area / max(frame_area, 1) <= self.config.max_flash_fraction
                )
                boxes = self._rise_boxes
                self._rise_frame = None
                self._rise_mask = None
                self._rise_boxes = ()
        self._previous = frame.copy()
        self._active = pulse
        self._cooldown = max(0, self._cooldown - 1)
        confidence = min(
            1.0,
            0.25 * min(changed_fraction / 0.01, 1.0)
            + 0.25 * min(candidate_fraction / 0.01, 1.0)
            + 0.20 * min(bright_fraction / 0.02, 1.0)
            + 0.30 * (1.0 if pulse else 0.0),
        )
        boxes = self._scale_boxes(boxes, scale)
        return self._result(
            self._active,
            confidence,
            changed_fraction,
            bright_fraction,
            boxes,
        )

    def _roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        if self._processing_roi is None:
            return np.ones(shape, dtype=bool)
        x, y, width, height = self._processing_roi
        if width <= 0 or height <= 0:
            raise ValueError("roi width and height must be positive")
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(shape[1], x + width), min(shape[0], y + height)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("roi does not overlap the frame")
        mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _boxes(mask: np.ndarray, min_area: int) -> tuple[tuple[int, int, int, int], ...]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return tuple(
            (int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
             int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= min_area
        )

    @staticmethod
    def _components(
        mask: np.ndarray,
        value: np.ndarray,
        value_rise: np.ndarray,
        min_area: int,
        min_brightness: int,
    ) -> tuple[_FlashComponent, ...]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: list[_FlashComponent] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component_mask = labels == index
            bright_pixels = component_mask & (value >= min_brightness)
            if not np.any(bright_pixels):
                continue
            mean_value = float(value[bright_pixels].mean())
            if mean_value < min_brightness:
                continue
            box = (
                int(stats[index, cv2.CC_STAT_LEFT]),
                int(stats[index, cv2.CC_STAT_TOP]),
                int(stats[index, cv2.CC_STAT_WIDTH]),
                int(stats[index, cv2.CC_STAT_HEIGHT]),
            )
            components.append(
                _FlashComponent(
                    box=box,
                    area=area,
                    mean_value=mean_value,
                    max_rise=int(value_rise[component_mask].max()),
                    center=(
                        float(stats[index, cv2.CC_STAT_LEFT])
                        + float(stats[index, cv2.CC_STAT_WIDTH]) / 2.0,
                        float(stats[index, cv2.CC_STAT_TOP])
                        + float(stats[index, cv2.CC_STAT_HEIGHT]) / 2.0,
                    ),
                )
            )
        return tuple(components)

    def _select_components(
        self, components: tuple[_FlashComponent, ...]
    ) -> tuple[_FlashComponent, ...]:
        if not components:
            return ()
        best_cluster: tuple[_FlashComponent, ...] = ()
        for component in components:
            cluster = tuple(
                other
                for other in components
                if (
                    (other.center[0] - component.center[0]) ** 2
                    + (other.center[1] - component.center[1]) ** 2
                    <= self.config.cluster_radius**2
                )
            )
            if sum(item.area for item in cluster) > sum(
                item.area for item in best_cluster
            ):
                best_cluster = cluster
        cluster_area = sum(item.area for item in best_cluster)
        cluster_brightness = (
            sum(item.area * item.mean_value for item in best_cluster)
            / max(cluster_area, 1)
        )
        if (
            cluster_area >= self.config.min_cluster_area
            and cluster_brightness >= self.config.min_cluster_brightness
        ):
            return best_cluster
        return tuple(
            component
            for component in components
            if component.max_rise >= self.config.min_strong_rise
        )

    @staticmethod
    def _component_mask(
        mask: np.ndarray,
        components: tuple[_FlashComponent, ...],
    ) -> np.ndarray:
        if not components:
            return np.zeros_like(mask)
        selected = np.zeros_like(mask)
        for x, y, width, height in (component.box for component in components):
            selected[y : y + height, x : x + width] |= mask[
                y : y + height, x : x + width
            ]
        return selected

    @staticmethod
    def _scale_boxes(
        boxes: tuple[tuple[int, int, int, int], ...], scale: float
    ) -> tuple[tuple[int, int, int, int], ...]:
        if scale >= 1.0:
            return boxes
        inverse = 1.0 / scale
        return tuple(
            (
                round(x * inverse),
                round(y * inverse),
                max(1, round(width * inverse)),
                max(1, round(height * inverse)),
            )
            for x, y, width, height in boxes
        )

    @staticmethod
    def _compactness(mask: np.ndarray) -> float:
        """Estimate whether the transient is a compact spark, not a diffuse wash."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return float(4.0 * np.pi * area / (perimeter * perimeter))

    def should_alert(self, detection: Detection) -> bool:
        """Return one alert for each distinct brightening/falling pulse."""
        if not detection.detected:
            self._alerted = False
            return False
        if self._alerted or self._cooldown:
            return False
        self._alerted = True
        self._cooldown = self.config.cooldown_frames
        return True

    @staticmethod
    def _result(
        detected: bool,
        confidence: float,
        changed_fraction: float,
        bright_fraction: float,
        boxes: tuple[tuple[int, int, int, int], ...],
    ) -> Detection:
        return Detection(
            detected=detected,
            confidence=confidence,
            changed_fraction=changed_fraction,
            bright_fraction=bright_fraction,
            boxes=boxes,
            message="SHORT/ARC DETECTED" if detected else "Normal",
        )
