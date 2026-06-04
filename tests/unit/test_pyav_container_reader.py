"""
Unit tests for PyAVContainerReader.

These tests work directly with PyAVContainerReader and a fake container —
no av.open, no device names, no format strings.  Tests for PyAVWebcamSource
(probing, av.open arguments, stream selection) live in
test_pyav_webcam_video_source.py.
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from unittest.mock import MagicMock

import numpy as np
import pytest

from py3r.media.video.pyav_webcam_source import PyAVContainerReader

W, H = 64, 48
FPS = 30.0


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

    ``streams`` is a list containing one MagicMock video stream so that
    ``_select_video_stream`` works without a separate stream stub.
    """

    def __init__(self, av_frames: list[FakeAVFrame], *,
                 error: Exception | None = None,
                 block_until_closed: bool = False):
        self._frames = av_frames
        self._error = error
        self._block_until_closed = block_until_closed
        self._closed = threading.Event()
        self.all_frames_yielded = threading.Event()

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
# Helpers
# ---------------------------------------------------------------------------

def _mock_stream() -> MagicMock:
    s = MagicMock()
    s.type = "video"
    return s


def _make_reader(av_frames: list[FakeAVFrame], *,
                 grayscale: bool = True,
                 queue_size: int = 8,
                 error: Exception | None = None,
                 block_until_closed: bool = False,
                 start_index: int = 0):
    """Return (reader, container)."""
    container = FakeContainer(av_frames, error=error,
                              block_until_closed=block_until_closed)
    reader = PyAVContainerReader(
        container, _mock_stream(),
        grayscale=grayscale,
        queue_size=queue_size,
        start_index=start_index,
    )
    return reader, container


def _gray(val: int = 128) -> np.ndarray:
    return np.full((H, W), val, dtype=np.uint8)


def _color(val: int = 128) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = val
    return img


# ---------------------------------------------------------------------------
# Frame delivery
# ---------------------------------------------------------------------------

class TestFrameDelivery:
    def test_frames_arrive_in_order(self):
        av_frames = [FakeAVFrame(_gray(i * 50), time=i / FPS) for i in range(5)]
        reader, container = _make_reader(av_frames)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frames = [reader.read(timeout=2.0) for _ in range(5)]
        reader.close()
        assert [f.frame_index for f in frames] == list(range(5))

    def test_frame_content_matches_input(self):
        expected = _gray(200)
        reader, container = _make_reader([FakeAVFrame(expected, time=0.0)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        np.testing.assert_array_equal(frame.img, expected)

    def test_grayscale_shape_and_dtype(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=0.0)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.img.shape == (H, W) and frame.img.dtype == np.uint8

    def test_color_shape_and_dtype(self):
        reader, container = _make_reader([FakeAVFrame(_color(), time=0.0)],
                                         grayscale=False)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.img.shape == (H, W, 3) and frame.img.dtype == np.uint8

    def test_timestamp_from_frame_time(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=1.234)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.timestamp == pytest.approx(1.234)

    def test_timestamp_pts_fallback(self):
        """frame.time=None → pts * time_base is used."""
        av_frame = FakeAVFrame(_gray(), time=None, pts=45, time_base=Fraction(1, 30))
        reader, container = _make_reader([av_frame])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.timestamp == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# End-of-stream
# ---------------------------------------------------------------------------

class TestEndOfStream:
    def test_eof_returns_none(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=0.0)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        reader.read(timeout=2.0)
        time.sleep(0.1)
        assert reader.read(timeout=0.5) is None
        reader.close()

    def test_read_after_eof_keeps_returning_none(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=0.0)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        reader.read(timeout=2.0)
        time.sleep(0.1)
        assert reader.read(timeout=0.2) is None
        assert reader.read(timeout=0.2) is None
        reader.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_reader_exception_raised_to_caller(self):
        boom = ValueError("simulated av error")
        reader, container = _make_reader([], error=boom)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        time.sleep(0.1)
        with pytest.raises(RuntimeError, match="reader failed"):
            reader.read(timeout=2.0)
        reader.close()

    def test_original_exception_is_chained(self):
        boom = ValueError("original cause")
        reader, container = _make_reader([], error=boom)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        time.sleep(0.1)
        with pytest.raises(RuntimeError) as exc_info:
            reader.read(timeout=2.0)
        reader.close()
        assert exc_info.value.__cause__ is boom


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_short_timeout_raises(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()
        with pytest.raises(TimeoutError):
            reader.read(timeout=0.05)
        reader.close()

    def test_generous_timeout_does_not_raise_if_frame_arrives(self):
        reader, _ = _make_reader([FakeAVFrame(_gray(), time=0.0)])
        reader.open()
        frame = reader.read(timeout=3.0)
        reader.close()
        assert frame is not None


# ---------------------------------------------------------------------------
# close() unblocks read()
# ---------------------------------------------------------------------------

class TestCloseUnblocks:
    def test_close_from_other_thread_unblocks_read(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()

        result_holder: list = []
        exc_holder: list = []

        def _reader():
            try:
                result_holder.append(reader.read(timeout=10.0))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_reader)
        t.start()
        time.sleep(0.1)
        reader.close()
        t.join(timeout=3.0)

        assert not t.is_alive(), "read() did not unblock after close()"
        assert not exc_holder
        assert result_holder == [None]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_open_before_open(self):
        reader, _ = _make_reader([])
        assert not reader.is_open()

    def test_is_open_after_open(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()
        assert reader.is_open()
        reader.close()

    def test_is_closed_after_close(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()
        reader.close()
        assert not reader.is_open()

    def test_double_close_is_safe(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()
        reader.close()
        reader.close()

    def test_frame_index_starts_at_zero(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=0.0)])
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.frame_index == 0

    def test_start_index_is_respected(self):
        reader, container = _make_reader([FakeAVFrame(_gray(), time=0.0)],
                                          start_index=42)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=3.0)
        frame = reader.read(timeout=2.0)
        reader.close()
        assert frame.frame_index == 42

    def test_open_is_idempotent(self):
        reader, _ = _make_reader([], block_until_closed=True)
        reader.open()
        thread_before = reader._reader_thread
        reader.open()
        assert reader._reader_thread is thread_before
        reader.close()


# ---------------------------------------------------------------------------
# Queue saturation
# ---------------------------------------------------------------------------

class TestQueueSaturation:
    def test_oldest_frames_dropped_when_full(self):
        N, QUEUE_SIZE = 12, 3
        av_frames = [FakeAVFrame(_gray(i * 20), time=i / FPS) for i in range(N)]
        reader, container = _make_reader(av_frames, queue_size=QUEUE_SIZE)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=5.0)
        time.sleep(0.05)

        received = []
        while True:
            f = reader.read(timeout=0.2)
            if f is None:
                break
            received.append(f)
        reader.close()

        assert len(received) <= QUEUE_SIZE
        assert received[-1].frame_index == N - 1

    def test_no_deadlock_under_saturation(self):
        av_frames = [FakeAVFrame(_gray(), time=i / FPS) for i in range(50)]
        reader, container = _make_reader(av_frames, queue_size=2)
        reader.open()
        assert container.all_frames_yielded.wait(timeout=5.0)
        reader.close()

