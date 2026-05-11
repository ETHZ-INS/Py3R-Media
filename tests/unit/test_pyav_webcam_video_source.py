"""
Unit tests for PyAVWebcamSource — no hardware required.

Fake objects
------------
FakeAVFrame
    Minimal duck-type for av.VideoFrame.  Only the three attributes used
    by _reader_loop and _frame_timestamp_seconds are implemented.

FakeContainer
    Drives the reader thread with a list of frames (or blocks forever).
    Uses MagicMock for .streams so that _select_video_stream works without
    a separate FakeStream class.

Factory pattern
---------------
_make_factory() returns a spy factory callable and a 'calls' list.
Each call to the factory creates a fresh FakeContainer, records the
(file, format, mode, options) arguments, and appends the result to calls.

    calls[0]  — probe container (opened and close()d by _probe())
    calls[1]  — live container  (created by open(); tests inspect this one)

_make_source() is a convenience wrapper for tests that don't need to
inspect factory arguments directly.
"""

from __future__ import annotations

import sys
import threading
import time
from fractions import Fraction
from unittest.mock import MagicMock

import numpy as np
import pytest

from py3r.media.video.pyav_webcam_source import PyAVWebcamSource

W, H = 64, 48
FPS = 30.0
DEVICE = "MockCamera"
EXPECTED_FORMAT = "dshow" if sys.platform == "win32" else (
    "avfoundation" if sys.platform == "darwin" else "v4l2"
)


# ---------------------------------------------------------------------------
# Fake PyAV objects
# ---------------------------------------------------------------------------

class FakeAVFrame:
    """Duck-type for av.VideoFrame.  Only the surface used by _reader_loop."""

    def __init__(self, img: np.ndarray, *, time: float | None = 0.0,
                 pts: int | None = None, time_base: Fraction | None = None):
        self.img = img
        self.time = time
        self.pts = pts
        self.time_base = time_base

    def to_ndarray(self, format: str) -> np.ndarray:  # noqa: A002
        return self.img.copy()


class FakeContainer:
    """
    Produces frames from a list, optionally blocking or raising an error.

    .streams is a MagicMock list so _select_video_stream works without a
    separate stream stub class.
    """

    def __init__(self, av_frames: list[FakeAVFrame], *,
                 error: Exception | None = None,
                 block_until_closed: bool = False):
        self._frames = av_frames
        self._error = error
        self._block_until_closed = block_until_closed
        self._closed = threading.Event()
        self.all_frames_yielded = threading.Event()

        # MagicMock stream: attribute access/assignment works transparently.
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.width = W
        mock_stream.codec_context.height = H
        mock_stream.average_rate = FPS
        self.streams = [mock_stream]

    def decode(self, stream):  # noqa: ARG002
        if self._block_until_closed:
            self._closed.wait()
            return
        yield from self._frames
        self.all_frames_yielded.set()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._closed.set()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_factory(av_frames: list[FakeAVFrame], *,
                  error: Exception | None = None,
                  block_until_closed: bool = False):
    """
    Return (factory, calls).

    calls is a list of dicts:
        {"file": str, "format": str, "mode": str,
         "options": dict, "container": FakeContainer}
    """
    calls: list[dict] = []

    def factory(file: str, *, format: str, mode: str, options: dict):  # noqa: A002
        container = FakeContainer(av_frames, error=error,
                                  block_until_closed=block_until_closed)
        calls.append({"file": file, "format": format, "mode": mode,
                       "options": dict(options), "container": container})
        return container

    return factory, calls


def _make_source(av_frames: list[FakeAVFrame], *,
                 grayscale: bool = True, queue_size: int = 8,
                 error: Exception | None = None,
                 block_until_closed: bool = False):
    """Convenience wrapper. Returns (src, calls)."""
    factory, calls = _make_factory(av_frames, error=error,
                                   block_until_closed=block_until_closed)
    src = PyAVWebcamSource(
        DEVICE,
        grayscale=grayscale,
        width=W, height=H, fps=FPS,
        queue_size=queue_size,
        container_factory=factory,
    )
    # calls[0] = probe container (already closed).  calls[1] = after open().
    return src, calls


def _gray(val: int = 128) -> np.ndarray:
    return np.full((H, W), val, dtype=np.uint8)


def _color(val: int = 128) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = val
    return img


def _live(calls: list[dict]) -> FakeContainer:
    """Return the container created by the most recent open() call."""
    return calls[-1]["container"]


# ---------------------------------------------------------------------------
# Container factory arguments
# ---------------------------------------------------------------------------

class TestContainerFactoryArgs:
    """Verify that _open_container passes the right arguments to av.open."""

    def _args(self, **kwargs) -> dict:
        """Construct source with given kwargs, return the probe call record."""
        factory, calls = _make_factory([])
        PyAVWebcamSource(DEVICE, container_factory=factory, **kwargs)
        return calls[0]  # probe call

    def test_file_is_device_name(self):
        rec = self._args()
        # dshow requires "video=<name>"; other platforms use the name directly.
        if rec["format"] == "dshow":
            assert rec["file"] == f"video={DEVICE}"
        else:
            assert rec["file"] == DEVICE

    def test_format_is_platform_default(self):
        assert self._args()["format"] == EXPECTED_FORMAT

    def test_explicit_format_is_forwarded(self):
        factory, calls = _make_factory([])
        PyAVWebcamSource(DEVICE, input_format="v4l2", container_factory=factory)
        assert calls[0]["format"] == "v4l2"

    def test_mode_is_read(self):
        assert self._args()["mode"] == "r"

    def test_options_video_size(self):
        opts = self._args(width=1280, height=720)["options"]
        assert opts["video_size"] == "1280x720"

    def test_options_framerate(self):
        opts = self._args(fps=60.0)["options"]
        assert opts["framerate"] == "60.0"

    def test_options_rtbufsize_on_dshow(self):
        factory, calls = _make_factory([])
        PyAVWebcamSource(DEVICE, input_format="dshow", container_factory=factory)
        assert calls[0]["options"].get("rtbufsize") == "500M"

    def test_no_rtbufsize_on_v4l2(self):
        factory, calls = _make_factory([])
        PyAVWebcamSource(DEVICE, input_format="v4l2", container_factory=factory)
        assert "rtbufsize" not in calls[0]["options"]

    def test_no_size_options_when_not_specified(self):
        opts = self._args()["options"]
        assert "video_size" not in opts

    def test_no_framerate_option_when_not_specified(self):
        opts = self._args()["options"]
        assert "framerate" not in opts

    def test_probe_and_open_receive_identical_args(self):
        """Both probe and live session should pass identical av.open arguments."""
        factory, calls = _make_factory([], block_until_closed=True)
        src = PyAVWebcamSource(DEVICE, container_factory=factory,
                                width=W, height=H, fps=FPS)
        src.open()
        src.close()
        assert len(calls) == 2
        probe, live = calls[0], calls[1]
        for key in ("file", "format", "mode", "options"):
            assert probe[key] == live[key], f"Mismatch on {key!r}"

    def test_raises_without_device_name(self):
        with pytest.raises(TypeError):
            PyAVWebcamSource()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Frame delivery
# ---------------------------------------------------------------------------

class TestFrameDelivery:
    def test_frames_arrive_in_order(self):
        av_frames = [FakeAVFrame(_gray(i * 50), time=i / FPS) for i in range(5)]
        src, calls = _make_source(av_frames)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frames = [src.read(timeout=2.0) for _ in range(5)]
        src.close()
        assert [f.frame_index for f in frames] == list(range(5))

    def test_frame_content_matches_input(self):
        expected = _gray(200)
        src, calls = _make_source([FakeAVFrame(expected, time=0.0)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        np.testing.assert_array_equal(frame.img, expected)

    def test_grayscale_shape_and_dtype(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=0.0)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.img.shape == (H, W) and frame.img.dtype == np.uint8

    def test_color_shape_and_dtype(self):
        src, calls = _make_source([FakeAVFrame(_color(), time=0.0)], grayscale=False)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.img.shape == (H, W, 3) and frame.img.dtype == np.uint8

    def test_timestamp_from_frame_time(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=1.234)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.timestamp == pytest.approx(1.234)

    def test_timestamp_pts_fallback(self):
        """frame.time=None → pts * time_base is used."""
        av_frame = FakeAVFrame(_gray(), time=None, pts=45, time_base=Fraction(1, 30))
        src, calls = _make_source([av_frame])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.timestamp == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# End-of-stream
# ---------------------------------------------------------------------------

class TestEndOfStream:
    def test_eof_returns_none(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=0.0)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        src.read(timeout=2.0)
        time.sleep(0.1)
        assert src.read(timeout=0.5) is None
        src.close()

    def test_read_after_eof_keeps_returning_none(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=0.0)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        src.read(timeout=2.0)
        time.sleep(0.1)
        assert src.read(timeout=0.2) is None
        assert src.read(timeout=0.2) is None
        src.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_reader_exception_raised_to_caller(self):
        boom = ValueError("simulated av error")
        src, calls = _make_source([], error=boom)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        time.sleep(0.1)
        with pytest.raises(RuntimeError, match="reader failed"):
            src.read(timeout=2.0)
        src.close()

    def test_original_exception_is_chained(self):
        boom = ValueError("original cause")
        src, calls = _make_source([], error=boom)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        time.sleep(0.1)
        with pytest.raises(RuntimeError) as exc_info:
            src.read(timeout=2.0)
        src.close()
        assert exc_info.value.__cause__ is boom


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_short_timeout_raises(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()
        with pytest.raises(TimeoutError):
            src.read(timeout=0.05)
        src.close()

    def test_generous_timeout_does_not_raise_if_frame_arrives(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=0.0)])
        src.open()
        frame = src.read(timeout=3.0)
        src.close()
        assert frame is not None


# ---------------------------------------------------------------------------
# close() unblocks read()
# ---------------------------------------------------------------------------

class TestCloseUnblocks:
    def test_close_from_other_thread_unblocks_read(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()

        result_holder: list = []
        exc_holder: list = []

        def _reader():
            try:
                result_holder.append(src.read(timeout=10.0))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_reader)
        t.start()
        time.sleep(0.1)
        src.close()
        t.join(timeout=3.0)

        assert not t.is_alive(), "read() did not unblock after close()"
        assert not exc_holder, f"Unexpected exception: {exc_holder}"
        assert result_holder == [None]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_open_before_open(self):
        src, _ = _make_source([])
        assert not src.is_open()

    def test_is_open_after_open(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()
        assert src.is_open()
        src.close()

    def test_is_closed_after_close(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()
        src.close()
        src.close()

    def test_frame_index_starts_at_zero(self):
        src, calls = _make_source([FakeAVFrame(_gray(), time=0.0)])
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.frame_index == 0

    def test_frame_index_resets_on_reopen(self):
        """After close() + open(), _idx must restart from 0."""
        factory, calls = _make_factory(
            [FakeAVFrame(_gray(i * 80), time=i / FPS) for i in range(3)]
        )
        src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS,
                                container_factory=factory)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        for _ in range(3):
            src.read(timeout=2.0)
        src.close()

        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=3.0)
        frame = src.read(timeout=2.0)
        src.close()
        assert frame.frame_index == 0

    def test_open_is_idempotent(self):
        src, _ = _make_source([], block_until_closed=True)
        src.open()
        thread_before = src._reader_thread
        src.open()
        assert src._reader_thread is thread_before
        src.close()


# ---------------------------------------------------------------------------
# Queue saturation
# ---------------------------------------------------------------------------

class TestQueueSaturation:
    def test_oldest_frames_dropped_when_full(self):
        N, QUEUE_SIZE = 12, 3
        av_frames = [FakeAVFrame(_gray(i * 20), time=i / FPS) for i in range(N)]
        src, calls = _make_source(av_frames, queue_size=QUEUE_SIZE)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=5.0)
        time.sleep(0.05)

        received = []
        while True:
            f = src.read(timeout=0.2)
            if f is None:
                break
            received.append(f)
        src.close()

        assert len(received) <= QUEUE_SIZE
        assert received[-1].frame_index == N - 1

    def test_no_deadlock_under_saturation(self):
        av_frames = [FakeAVFrame(_gray(), time=i / FPS) for i in range(50)]
        src, calls = _make_source(av_frames, queue_size=2)
        src.open()
        assert _live(calls).all_frames_yielded.wait(timeout=5.0)
        src.close()



