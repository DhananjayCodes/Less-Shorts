import cv2
import numpy as np

from short_detector.detector import DetectorConfig, ShortCircuitDetector


def frame_with_square(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (60, 60), color, -1)
    return frame


def test_static_bright_region_is_not_an_event() -> None:
    detector = ShortCircuitDetector(DetectorConfig(event_frames=1))
    frame = frame_with_square((255, 255, 255))
    detector.process(frame)
    result = detector.process(frame)
    assert not result.detected


def test_bright_object_outside_roi_is_ignored() -> None:
    detector = ShortCircuitDetector(
        DetectorConfig(event_frames=1, min_area=4, min_flash_area=4, roi=(0, 0, 30, 30))
    )
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    white = dark.copy()
    cv2.rectangle(white, (70, 70), (90, 90), (255, 255, 255), -1)
    detector.process(dark)
    detector.process(white)
    assert not detector.process(dark).detected


def test_large_scene_flash_is_rejected() -> None:
    detector = ShortCircuitDetector(
        DetectorConfig(min_area=4, min_flash_area=4, max_flash_fraction=0.02)
    )
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    bright = np.full((100, 100, 3), 255, dtype=np.uint8)
    detector.process(dark)
    detector.process(bright)
    assert not detector.process(dark).detected


def test_bright_color_change_triggers_after_persistence() -> None:
    detector = ShortCircuitDetector(
        DetectorConfig(
            min_area=4, min_flash_area=4, bright_value=220, max_flash_fraction=0.1
        )
    )
    dark = frame_with_square((0, 0, 0))
    first_flash = frame_with_square((220, 220, 220))
    detector.process(dark)
    assert not detector.process(first_flash).detected
    result = detector.process(dark)
    assert result.detected
    assert result.boxes
    assert detector.should_alert(result)
    assert not detector.should_alert(result)


def test_small_colored_flash_triggers() -> None:
    detector = ShortCircuitDetector()
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    flash = dark.copy()
    cv2.rectangle(flash, (49, 49), (50, 50), (255, 0, 0), -1)
    detector.process(dark)
    detector.process(flash)
    result = detector.process(dark)
    assert result.detected
    assert result.boxes


def test_diffuse_sky_like_change_is_rejected() -> None:
    detector = ShortCircuitDetector(
        DetectorConfig(min_area=2, min_flash_area=2, max_flash_fraction=0.04)
    )
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    cloud = dark.copy()
    cv2.ellipse(cloud, (50, 50), (35, 25), 0, 0, 360, (230, 230, 230), -1)
    detector.process(dark)
    detector.process(cloud)
    assert not detector.process(dark).detected


def test_clustered_arc_filaments_trigger() -> None:
    detector = ShortCircuitDetector()
    dark = np.zeros((120, 120, 3), dtype=np.uint8)
    flash = dark.copy()
    for x, y in ((48, 48), (56, 49), (64, 50), (52, 57), (60, 59), (68, 57)):
        cv2.rectangle(flash, (x, y), (x + 4, y + 4), (255, 255, 255), -1)
    detector.process(dark)
    detector.process(flash)
    result = detector.process(dark)
    assert result.detected
    assert result.boxes


def test_invalid_frame_is_rejected() -> None:
    detector = ShortCircuitDetector()
    with np.testing.assert_raises(ValueError):
        detector.process(np.zeros((10, 10), dtype=np.uint8))


def test_720p_frames_are_processed() -> None:
    detector = ShortCircuitDetector(
        DetectorConfig(process_width=640, min_area=4, min_flash_area=4)
    )
    dark = np.zeros((720, 1280, 3), dtype=np.uint8)
    flash = dark.copy()
    cv2.rectangle(flash, (600, 340), (630, 370), (255, 255, 255), -1)
    detector.process(dark)
    detector.process(flash)
    result = detector.process(dark)
    assert result.boxes
    assert result.detected
    x, y, width, height = result.boxes[0]
    assert 590 <= x <= 610
    assert 330 <= y <= 350
    assert width >= 30
    assert height >= 30
