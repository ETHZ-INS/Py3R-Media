"""
Unit tests for OpenCVWebcamSource — no hardware required.

Fake objects
------------
FakeCapture
    Duck-type for cv2.VideoCapture.  Stores a list of BGR frames and serves
    them via read(); get() returns configured width/height/fps metadata.

Factory pattern
---------------
_make_factory() returns a spy factory and a 'calls' list.  Each call to the
factory creates a fresh FakeCapture and appends {"device", "backend",
"capture"} to calls.

    calls[0]  — probe capture (created and released() by _probe())
    calls[1]  — live capture  (created by open())

_make_source() is a convenience wrapper for tests that don't need to inspect
factory arguments directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from py3r.media.types import VideoFrame
from py3r.media.video.opencv_webcam_source import OpenCVWebcamSource, ReadFailedError

W, H = 64, 48
FPS = 30.0
DEVICE = 7          # arbitrary integer device index
BACKEND = cv2.CAP_ANY


# ---------------------------------------------------------------------------
# Fake cv2.VideoCapture
# ---------------------------------------------------------------------------

class FakeCapture:
    """Minimal duck-type for cv2.VideoCapture."""

    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        width: int = W,
        height: int = H,
        fps: float = FPS,
        opened: bool = True,
    ):
        self._frames = list(frames)
        self._width = width
        self._height = height
        self._fps = fps
        self._opened = opened
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._height)
        if prop_id == cv2.CAP_PROP_FPS:          return float(self._fps)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0).copy()

    def release(self) -> None:
        self.released = True
        self._opened = False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _bgr_frame(val: int = 128) -> np.ndarray:
    return np.full((H, W, 3), val, dtype=np.uint8)


def _bgr_frames(n: int, start: int = 0) -> list[np.ndarray]:
    return [_bgr_frame((start + i) % 256) for i in range(n)]


def _make_factory(
    frames: list[np.ndarray],
    *,
    width: int = W,
    height: int = H,
    fps: float = FPS,
    opened: bool = True,
):
    """Return (factory, calls).  Each factory call creates a fresh FakeCapture."""
    calls: list[dict] = []

    def factory(device, *, backend: int):
        cap = FakeCapture(list(frames), width=width, height=height,
                          fps=fps, opened=opened)
        calls.append({"device": device, "backend": backend, "capture": cap})
        return cap

    return factory, calls


def _make_source(
    frames: list[np.ndarray],
    *,
    grayscale: bool = True,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    backend: int = BACKEND,
    cap_width: int = W,
    cap_height: int = H,
    cap_fps: float = FPS,
):
    """Convenience wrapper.  Returns (src, calls); calls[0]=probe, calls[1]=live."""
    factory, calls = _make_factory(frames, width=cap_width, height=cap_height,
                                   fps=cap_fps)
    src = OpenCVWebcamSource(
        DEVICE,
        grayscale=grayscale,
        width=width,
        height=height,
        fps=fps,
        backend=backend,
        capture_factory=factory,
    )
    return src, calls


# ---------------------------------------------------------------------------
# Helpers to get the live capture (index 1 after open())
# ---------------------------------------------------------------------------

def _live(calls: list[dict]) -> FakeCapture:
    return calls[1]["capture"]


# ---------------------------------------------------------------------------
# TestCaptureFactoryArgs
# ---------------------------------------------------------------------------

class TestCaptureFactoryArgs:
    def test_device_passed_to_factory(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(42, capture_factory=factory)
        assert calls[0]["device"] == 42

    def test_string_device_passed_to_factory(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource("/dev/video0", capture_factory=factory)
        assert calls[0]["device"] == "/dev/video0"

    def test_backend_passed_to_factory(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(0, backend=cv2.CAP_DSHOW, capture_factory=factory)
        assert calls[0]["backend"] == cv2.CAP_DSHOW

    def test_default_backend_is_cap_any(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(0, capture_factory=factory)
        assert calls[0]["backend"] == cv2.CAP_ANY

    def test_factory_called_once_for_probe(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(0, capture_factory=factory)
        assert len(calls) == 1

    def test_factory_called_again_for_open(self):
        src, calls = _make_source(_bgr_frames(1))
        src.open()
        assert len(calls) == 2

    def test_probe_capture_is_released(self):
        src, calls = _make_source(_bgr_frames(1))
        assert calls[0]["capture"].released

    def test_live_capture_not_released_while_open(self):
        src, calls = _make_source(_bgr_frames(1))
        src.open()
        assert not _live(calls).released

    def test_live_capture_released_after_close(self):
        src, calls = _make_source(_bgr_frames(1))
        src.open()
        src.close()
        assert _live(calls).released


# ---------------------------------------------------------------------------
# TestProbe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_populated_from_capture(self):
        src, _ = _make_source([], cap_width=320, cap_height=240)
        assert src.get_size() == (320, 240)

    def test_fps_populated_from_capture(self):
        src, _ = _make_source([], cap_fps=60.0)
        assert src.get_fps() == 60.0

    def test_has_size_true_after_probe(self):
        src, _ = _make_source([])
        assert src.has_size()

    def test_has_fps_true_after_probe(self):
        src, _ = _make_source([])
        assert src.has_fps()

    def test_probe_soft_fails_when_factory_raises(self):
        """Construction must not raise even if the factory fails."""
        def bad_factory(device, *, backend):
            raise RuntimeError("no camera")

        src = OpenCVWebcamSource(0, capture_factory=bad_factory)
        assert src.get_size() is None
        assert not src.has_size()

    def test_probe_soft_fails_when_not_opened(self):
        factory, _ = _make_factory([], opened=False)
        src = OpenCVWebcamSource(0, capture_factory=factory)
        assert src.get_size() is None

    def test_configure_called_with_requested_size(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(0, width=W, height=H, capture_factory=factory)
        probe_cap = calls[0]["capture"]
        prop_ids = [p for p, _ in probe_cap.set_calls]
        assert cv2.CAP_PROP_FRAME_WIDTH in prop_ids
        assert cv2.CAP_PROP_FRAME_HEIGHT in prop_ids

    def test_configure_called_with_requested_fps(self):
        factory, calls = _make_factory(_bgr_frames(1))
        OpenCVWebcamSource(0, fps=FPS, capture_factory=factory)
        probe_cap = calls[0]["capture"]
        prop_ids = [p for p, _ in probe_cap.set_calls]
        assert cv2.CAP_PROP_FPS in prop_ids


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_open_before_open(self):
        src, _ = _make_source([])
        assert not src.is_open()

    def test_is_open_after_open(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        assert src.is_open()

    def test_not_open_after_close(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        src.close()
        src.close()  # must not raise

    def test_capability_flags(self):
        src, _ = _make_source([])
        assert src.has_timing()
        assert not src.has_num_frames()
        assert not src.is_seekable()

    def test_num_channels_grayscale(self):
        src, _ = _make_source([], grayscale=True)
        assert src.get_num_channels() == 1

    def test_num_channels_color(self):
        src, _ = _make_source([], grayscale=False)
        assert src.get_num_channels() == 3


# ---------------------------------------------------------------------------
# TestFrameContent
# ---------------------------------------------------------------------------

class TestFrameContent:
    def test_read_returns_video_frame(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        frame = src.read()
        assert isinstance(frame, VideoFrame)

    def test_grayscale_frame_is_2d(self):
        src, _ = _make_source(_bgr_frames(1), grayscale=True)
        src.open()
        frame = src.read()
        assert frame.img.ndim == 2

    def test_grayscale_frame_dtype(self):
        src, _ = _make_source(_bgr_frames(1), grayscale=True)
        src.open()
        assert src.read().img.dtype == np.uint8

    def test_color_frame_is_3d(self):
        src, _ = _make_source(_bgr_frames(1), grayscale=False)
        src.open()
        frame = src.read()
        assert frame.img.ndim == 3
        assert frame.img.shape[2] == 3

    def test_color_frame_preserves_values(self):
        """BGR values should be passed through unchanged."""
        bgr = np.zeros((H, W, 3), dtype=np.uint8)
        bgr[..., 0] = 10   # blue
        bgr[..., 1] = 20   # green
        bgr[..., 2] = 30   # red
        factory, calls = _make_factory([bgr])
        src = OpenCVWebcamSource(DEVICE, grayscale=False, capture_factory=factory)
        src.open()
        out = src.read().img
        np.testing.assert_array_equal(out, bgr)

    def test_frame_size_matches_capture_size(self):
        src, _ = _make_source(_bgr_frames(1), grayscale=False)
        src.open()
        frame = src.read()
        h, w = frame.img.shape[:2]
        assert (w, h) == src.get_size()


# ---------------------------------------------------------------------------
# TestFrameIndex
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_first_frame_index_is_zero(self):
        src, _ = _make_source(_bgr_frames(3))
        src.open()
        assert src.read().frame_index == 0

    def test_frame_index_increments(self):
        src, _ = _make_source(_bgr_frames(3))
        src.open()
        indices = [src.read().frame_index for _ in range(3)]
        assert indices == [0, 1, 2]

    def test_frame_index_resets_on_reopen(self):
        factory, _ = _make_factory(_bgr_frames(3))
        src = OpenCVWebcamSource(DEVICE, capture_factory=factory)
        src.open()
        src.read()
        src.close()
        src.open()
        assert src.read().frame_index == 0


# ---------------------------------------------------------------------------
# TestTimestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamp_is_positive(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        assert src.read().timestamp > 0.0

    def test_timestamps_non_decreasing(self):
        src, _ = _make_source(_bgr_frames(5))
        src.open()
        ts = [src.read().timestamp for _ in range(5)]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


# ---------------------------------------------------------------------------
# TestReadFailed
# ---------------------------------------------------------------------------

class TestReadFailed:
    def test_read_failed_raises_read_failed_error(self):
        """FakeCapture with no frames returns ok=False → ReadFailedError."""
        src, _ = _make_source([])  # no frames
        src.open()
        with pytest.raises(ReadFailedError):
            src.read()

    def test_read_failed_message_contains_device(self):
        src, _ = _make_source([])
        src.open()
        with pytest.raises(ReadFailedError, match=str(DEVICE)):
            src.read()


# ---------------------------------------------------------------------------
# TestTimeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_expired_deadline_raises_timeout_error(self):
        """Pass timeout=0 so the deadline is already past before read()."""
        src, _ = _make_source([])
        src.open()
        with pytest.raises(TimeoutError):
            src.read(timeout=0.0)


# ---------------------------------------------------------------------------
# TestNotOpen
# ---------------------------------------------------------------------------

class TestNotOpen:
    def test_read_before_open_raises_runtime_error(self):
        src, _ = _make_source([])
        with pytest.raises(RuntimeError, match="not open"):
            src.read()

    def test_read_after_close_raises_runtime_error(self):
        src, _ = _make_source(_bgr_frames(1))
        src.open()
        src.close()
        with pytest.raises(RuntimeError, match="not open"):
            src.read()

