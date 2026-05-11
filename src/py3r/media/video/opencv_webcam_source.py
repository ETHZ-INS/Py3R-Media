import time
from typing import Any, Optional, Protocol, Tuple, Union

import cv2

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class ReadFailedError(OSError):
    """Raised when ``cv2.VideoCapture.read()`` returns ``ok=False``."""


class OpenCVWebcamSource(VideoSource):
    """
    Pull-based webcam source via OpenCV VideoCapture.

    Parameters
    ----------
    device : int | str
        Device identifier passed to ``cv2.VideoCapture``.

        * **Integer index** — ``0`` for the first camera, ``1`` for the second, etc.
        * **V4L2 path** — e.g. ``"/dev/video0"`` on Linux.
        * **GStreamer / URL pipeline** — any string accepted by OpenCV.
    grayscale : bool
        Convert frames to single-channel grayscale before returning them.
    width, height : int | None
        Request a specific capture resolution.  The device may ignore or round this.
    fps : float | None
        Request a specific frame-rate.
    backend : int
        OpenCV capture backend flag (e.g. ``cv2.CAP_DSHOW``).
        Defaults to ``cv2.CAP_ANY``.
    capture_factory : CaptureFactory | None
        If provided, replaces ``cv2.VideoCapture(device, backend)`` for both
        probe and live capture.  Receives the same arguments::

            capture_factory(device, backend=backend) -> capture

        This lets tests verify that the source passes the right device and
        backend, and return a fake capture to drive the rest of the pipeline.
    """

    class CaptureFactory(Protocol):
        def __call__(self, device: Union[int, str], *, backend: int) -> Any: ...

    def __init__(
        self,
        device: Union[int, str] = 0,
        *,
        grayscale: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        backend: int = cv2.CAP_ANY,
        capture_factory: Optional[CaptureFactory] = None,
    ):
        self._idx = 0
        self._cap = None
        self._device = device
        self._backend = backend
        self._grayscale = grayscale
        self._requested_size = (width, height) if width and height else None
        self._requested_fps = fps
        self._size: Optional[Tuple[int, int]] = None
        self._fps: Optional[float] = None
        self._capture_factory = capture_factory

        self._probe()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._cap = self._open_camera()
        self._configure_camera(self._cap)
        self._idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return self._size is not None
    def has_fps(self) -> bool: return self._fps is not None
    def has_num_frames(self) -> bool: return False
    def is_seekable(self) -> bool: return False

    def get_size(self) -> Optional[Tuple[int, int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return 1 if self._grayscale else 3
    def get_num_frames(self) -> Optional[int]: return None

    def seek(self, frame_index: int) -> None:
        pass  # live source; no-op

    def read(self, timeout: Optional[float] = None) -> VideoFrame:
        """
        Read the next frame from the camera.

        Raises
        ------
        RuntimeError
            If the source is not open.
        ReadFailedError
            If ``cv2.VideoCapture.read()`` returns ``ok=False``.
        TimeoutError
            If *timeout* seconds elapse without a successful read.
        """
        if self._cap is None:
            raise RuntimeError(
                f"Cannot read from device {self._device!r}: source is not open"
            )

        deadline = None if timeout is None else (time.perf_counter() + timeout)

        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"Timeout reading frame from device {self._device!r}"
                )

            ok, img = self._cap.read()

            if ok:
                ts = time.perf_counter()
                if self._grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame = VideoFrame(img, self._idx, ts)
                self._idx += 1
                return frame

            raise ReadFailedError(
                f"cv2.VideoCapture.read() failed for device {self._device!r}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture:
        if self._capture_factory is not None:
            cam = self._capture_factory(self._device, backend=self._backend)
        else:
            cam = cv2.VideoCapture(self._device, self._backend)
        if not cam.isOpened():
            raise RuntimeError(
                f"Cannot open webcam device {self._device!r} "
                f"(backend={self._backend})"
            )
        return cam

    def _configure_camera(self, camera: cv2.VideoCapture) -> None:
        if self._requested_size:
            w, h = self._requested_size
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if self._requested_fps:
            camera.set(cv2.CAP_PROP_FPS, self._requested_fps)

    def _probe(self) -> None:
        try:
            cam = self._open_camera()
            try:
                self._configure_camera(cam)
                w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w and h:
                    self._size = (w, h)
                raw_fps = cam.get(cv2.CAP_PROP_FPS)
                self._fps = float(raw_fps) if raw_fps else None
            finally:
                cam.release()
        except Exception:
            # Keep probing soft: caller can still open() later.
            pass
