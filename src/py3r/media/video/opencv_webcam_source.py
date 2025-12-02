import time
from typing import Optional, Tuple

import cv2

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class OpenCVWebcamSource(VideoSource):
    def __init__(self, device_index: int = 0, grayscale: bool = True,
                 width: Optional[int] = None, height: Optional[int] = None, fps: Optional[float] = None):
        self._idx = 0
        self._cap = None
        self._device = device_index
        self._grayscale = grayscale
        self._requested_size = (width, height) if width and height else None
        self._requested_fps = fps
        self._size = None
        self._fps = None

        self._probe()

    def open(self) -> None:
        self._cap = self._open_camera()
        self._configure_camera(self._cap)
        self._idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def is_open(self) -> bool: return self._cap is not None and self._cap.isOpened()
    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return True
    def has_fps(self) -> bool: return bool(self._fps)
    def has_num_frames(self) -> bool: return False
    def is_seekable(self) -> bool: return False

    def get_size(self) -> Optional[Tuple[int,int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return 1 if self._grayscale else 3
    def get_num_frames(self) -> Optional[int]: return None

    def seek(self, frame_index: int) -> None: pass

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        t0 = time.perf_counter()
        while True:
            ok, img = self._cap.read()
            if ok:
                ts = time.perf_counter()  # monotonic
                if self._grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f = VideoFrame(img, self._idx, ts)
                self._idx += 1
                return f
            if timeout is not None and (time.perf_counter() - t0) > timeout:
                return None

    def _open_camera(self) -> cv2.VideoCapture:
        cam = cv2.VideoCapture(self._device, cv2.CAP_ANY)
        if not cam.isOpened():
            raise RuntimeError("Cannot open webcam")
        return cam

    def _configure_camera(self, camera: cv2.VideoCapture) -> None:
        if self._requested_size:
            w,h = self._requested_size
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if self._requested_fps:
            camera.set(cv2.CAP_PROP_FPS, self._requested_fps)

    def _probe(self):
        cam = self._open_camera()
        self._configure_camera(cam)

        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._size = (w,h)
        self._fps = float(cam.get(cv2.CAP_PROP_FPS) or 0) or None
        cam.release()
