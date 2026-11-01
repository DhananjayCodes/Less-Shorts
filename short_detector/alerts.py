from __future__ import annotations

from datetime import datetime, timezone

from .detector import Detection


def console_alert(detection: Detection) -> None:
    """Emit a visible and audible terminal alert."""
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"\a[{timestamp}] ALERT: {detection.message} "
        f"(confidence={detection.confidence:.0%})",
        flush=True,
    )
