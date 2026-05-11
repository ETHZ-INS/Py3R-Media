"""
Hardware tests for PyAVWebcamSource.

These tests are skipped automatically unless ``--webcam-name`` is passed.
The flag accepts the platform device identifier for PyAV:

* **Windows DirectShow** — the display name, e.g. ``"Integrated Camera"``
* **Linux V4L2**         — the device node, e.g. ``"/dev/video0"``
* **macOS AVFoundation** — the device index as string, e.g. ``"0"``

Example::

    pytest tests/hardware/test_pyav_webcam_video_source.py \\
           --webcam-name="Integrated Camera" -m hardware

Pass ``--require-webcam`` to turn skips into failures (useful in CI).

Because opening and closing a webcam is slow (and may cause frame-drop
artefacts), a single session-scoped fixture opens the source once and most
tests share the same live stream.

Tests that need exclusive device access (color read, lifecycle open/close,
timeout) use the ``isolated_source`` function fixture, which temporarily
closes the session source, runs the test, then reopens it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from py3r.media.video.pyav_webcam_source import PyAVWebcamSource


# ---------------------------------------------------------------------------
# Mark every test in this module as a hardware test
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.hardware]


# ---------------------------------------------------------------------------
# Session-scoped grayscale source (kept open for the whole session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pyav_webcam_gray(pyav_webcam_name: str) -> PyAVWebcamSource:
    """
    Open a PyAVWebcamSource (grayscale) against the device given by
    ``--webcam-name``.  Skipped if construction fails.
    """
    try:
        src = PyAVWebcamSource(pyav_webcam_name, grayscale=True)
    except Exception as exc:
        pytest.skip(f"PyAVWebcamSource('{pyav_webcam_name}') failed to init: {exc}")

    src.open()
    time.sleep(0.5)   # let the background reader buffer a frame

    yield src

    src.close()


# ---------------------------------------------------------------------------
# Function fixture: temporarily close the session source so the test has
# exclusive access to the device, then restore it afterwards.
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_source(
    pyav_webcam_gray: PyAVWebcamSource,
    pyav_webcam_name: str,
):
    """
    Close the session gray source, yield the device name for the test to use,
    then reopen the session source.  Ensures no two sources are open at once.
    """
    pyav_webcam_gray.close()
    try:
        yield pyav_webcam_name
    finally:
        pyav_webcam_gray.open()
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Probe / metadata
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_is_populated(self, pyav_webcam_gray: PyAVWebcamSource):
        """Size must be known after construction (probe runs in __init__)."""
        size = pyav_webcam_gray.get_size()
        assert size is not None
        w, h = size
        assert w > 0 and h > 0

    def test_capability_flags(self, pyav_webcam_gray: PyAVWebcamSource):
        assert pyav_webcam_gray.has_timing()
        assert not pyav_webcam_gray.has_num_frames()
        assert not pyav_webcam_gray.is_seekable()

    def test_num_channels_grayscale(self, pyav_webcam_gray: PyAVWebcamSource):
        assert pyav_webcam_gray.get_num_channels() == 1

    def test_num_channels_color(self, isolated_source: str):
        """Probe-only — checks get_num_channels() without keeping device open."""
        try:
            src = PyAVWebcamSource(isolated_source, grayscale=False)
        except Exception as exc:
            pytest.skip(f"Could not init color source: {exc}")
        assert src.get_num_channels() == 3

    def test_fps_is_positive_if_known(self, pyav_webcam_gray: PyAVWebcamSource):
        fps = pyav_webcam_gray.get_fps()
        if fps is not None:
            assert fps > 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_open_after_open(self, pyav_webcam_gray: PyAVWebcamSource):
        assert pyav_webcam_gray.is_open()

    def test_close_stops_is_open(self, isolated_source: str):
        """Open and close a fresh source with exclusive device access."""
        try:
            src = PyAVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        assert src.is_open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self, isolated_source: str):
        try:
            src = PyAVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        src.close()
        src.close()  # must not raise


# ---------------------------------------------------------------------------
# Read — grayscale
# ---------------------------------------------------------------------------

class TestReadGrayscale:
    def test_read_returns_video_frame(self, pyav_webcam_gray: PyAVWebcamSource):
        from py3r.media.types import VideoFrame
        frame = pyav_webcam_gray.read(timeout=5.0)
        assert isinstance(frame, VideoFrame)

    def test_frame_shape_is_2d(self, pyav_webcam_gray: PyAVWebcamSource):
        """Grayscale frames must have shape (H, W)."""
        frame = pyav_webcam_gray.read(timeout=5.0)
        assert frame.img.ndim == 2

    def test_frame_dtype_is_uint8(self, pyav_webcam_gray: PyAVWebcamSource):
        frame = pyav_webcam_gray.read(timeout=5.0)
        assert frame.img.dtype == np.uint8

    def test_frame_size_matches_probe(self, pyav_webcam_gray: PyAVWebcamSource):
        size = pyav_webcam_gray.get_size()
        frame = pyav_webcam_gray.read(timeout=5.0)
        h, w = frame.img.shape
        assert (w, h) == size

    def test_frame_is_not_all_zeros(self, pyav_webcam_gray: PyAVWebcamSource):
        """A real camera frame must contain at least some non-zero pixels."""
        frame = pyav_webcam_gray.read(timeout=5.0)
        assert frame.img.max() > 0


# ---------------------------------------------------------------------------
# Read — color
# ---------------------------------------------------------------------------

@pytest.fixture()
def pyav_color_frame(isolated_source: str) -> np.ndarray:
    """Open a color source with exclusive access, grab one frame, close."""
    try:
        src = PyAVWebcamSource(isolated_source, grayscale=False)
        src.open()
        time.sleep(0.5)
        try:
            frame = src.read(timeout=5.0)
        finally:
            src.close()
    except Exception as exc:
        pytest.skip(f"Could not read color frame: {exc}")
    return frame.img


class TestReadColor:
    def test_frame_shape_is_3d(self, pyav_color_frame: np.ndarray):
        assert pyav_color_frame.ndim == 3
        assert pyav_color_frame.shape[2] == 3

    def test_frame_dtype_is_uint8(self, pyav_color_frame: np.ndarray):
        assert pyav_color_frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_frame_index_increments(self, pyav_webcam_gray: PyAVWebcamSource):
        f1 = pyav_webcam_gray.read(timeout=5.0)
        f2 = pyav_webcam_gray.read(timeout=5.0)
        assert f2.frame_index == f1.frame_index + 1

    def test_five_consecutive_indices(self, pyav_webcam_gray: PyAVWebcamSource):
        frames = [pyav_webcam_gray.read(timeout=5.0) for _ in range(5)]
        indices = [f.frame_index for f in frames]
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        assert all(d == 1 for d in diffs), f"Non-consecutive indices: {indices}"


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamp_is_non_negative(self, pyav_webcam_gray: PyAVWebcamSource):
        frame = pyav_webcam_gray.read(timeout=5.0)
        assert frame.timestamp >= 0.0

    def test_timestamps_are_non_decreasing(self, pyav_webcam_gray: PyAVWebcamSource):
        frames = [pyav_webcam_gray.read(timeout=5.0) for _ in range(5)]
        ts = [f.timestamp for f in frames]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_zero_timeout_raises_timeout_error(self, isolated_source: str):
        """
        Open a *fresh* source, do NOT wait for the reader to buffer a frame,
        then immediately request with timeout=0.  The queue should be empty
        and TimeoutError should be raised.
        """
        try:
            src = PyAVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")

        src.open()
        # Do NOT sleep — give the background thread no time to buffer a frame.
        try:
            with pytest.raises(TimeoutError):
                src.read(timeout=0.0)
        finally:
            src.close()
