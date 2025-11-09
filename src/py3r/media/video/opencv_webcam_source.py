import time
from typing import Optional, Tuple

import cv2

from py3r.media.types import VideoFrame


class OpenCVWebcamSource:
    def __init__(self, device_index: int = 0, grayscale: bool = True, width: int | None = None, height: int | None = None, fps: float | None = None):
        self._idx = 0
        self._cap = None
        self._device = device_index
        self._gray = grayscale
        self._requested_size = (width, height) if width and height else None
        self._requested_fps = fps
        self._size = None
        self._fps = None

    def name(self) -> str: return f"webcam:{self._device}"
    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        if self._requested_size:
            w,h = self._requested_size
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if self._requested_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # Probe actual
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._size = (w,h)
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0) or None
        self._idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def is_open(self) -> bool: return self._cap is not None and self._cap.isOpened()
    def has_timing(self) -> bool: return True
    def has_fixed_fps(self) -> bool: return bool(self._fps)
    def has_fixed_size(self) -> bool: return True
    def is_seekable(self) -> bool: return False

    def get_fps(self) -> Optional[float]: return self._fps
    def get_size(self) -> Optional[Tuple[int,int]]: return self._size
    def get_num_channels(self) -> int: return 1 if self._gray else 3

    def set_playback_rate(self, mode: str) -> None: pass
    def seek(self, frame_index: int) -> None: pass
    def enable_grayscale(self, gray: bool) -> None: self._gray = gray

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        t0 = time.perf_counter()
        while True:
            ok, img = self._cap.read()
            if ok:
                ts = time.perf_counter()  # monotonic
                if self._gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f = VideoFrame(img, self._idx, ts)
                self._idx += 1
                return f
            if timeout is not None and (time.perf_counter() - t0) > timeout:
                return None
