"""A wrapper around screeninfo with improved functionality on macOS."""
import sys
from typing import Iterable, List, Optional, Tuple

import screeninfo
from screeninfo import Monitor


def get_monitors() -> List[Monitor]:
    """Return a list of Monitor objects representing the connected screens."""
    if sys.platform == "darwin":
        return list(_enumerate_macos_monitors())
    return screeninfo.get_monitors()


def get_ppmm(monitor: Monitor) -> Optional[Tuple[float, float]]:
    """Return the pixels per millimeter of the given monitor."""
    if not monitor.width_mm or not monitor.height_mm:
        return None
    return (monitor.width / monitor.width_mm, monitor.height / monitor.height_mm)


def _enumerate_macos_monitors() -> Iterable[Monitor]:
    from AppKit import NSDeviceSize  # pyright: ignore [reportMissingImports]
    from AppKit import NSScreen  # pyright: ignore [reportMissingImports]
    from Quartz import CGDisplayScreenSize  # pyright: ignore [reportMissingImports]

    screens = NSScreen.screens()
    for screen in screens:
        f = screen.frame
        if callable(f):
            f = f()

        description = screen.deviceDescription()
        width, height = description[NSDeviceSize].sizeValue()
        width_mm, height_mm = CGDisplayScreenSize(description["NSScreenNumber"])
        yield Monitor(
            x=int(f.origin.x),
            y=int(f.origin.y),
            width=width,
            height=height,
            width_mm=width_mm,
            height_mm=height_mm,
            is_primary=(screen == screens[0]),
        )
