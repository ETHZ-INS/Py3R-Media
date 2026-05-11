"""
Hardware tests for OpenCVWebcamSource.

Skipped automatically when no USB webcam is detected.  Pass
``--require-webcam`` to turn skips into failures (useful in CI).

A session-scoped fixture opens the grayscale source once; all grayscale tests
share that live stream.  Color tests open their own short-lived source because
most webcam drivers (e.g. DirectShow on Windows) only allow one VideoCapture
consumer at a time — having two session sources open simultaneously causes
read() to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from py3r.media.types import VideoFrame
from py3r.media.video.opencv_webcam_source import OpenCVWebcamSource, ReadFailedError


pytestmark = [pytest.mark.hardware]


# ---------------------------------------------------------------------------
# Session-scoped grayscale source  (kept open for the whole session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cv_webcam_gray(first_webcam_index: int) -> OpenCVWebcamSource:
    """Open an OpenCVWebcamSource (grayscale) on the first available device."""
    try:
        src = OpenCVWebcamSource(first_webcam_index, grayscale=True)
    except Exception as exc:
        pytest.skip(f"OpenCVWebcamSource({first_webcam_index}) failed to init: {exc}")

    src.open()
    yield src
    src.close()


# ---------------------------------------------------------------------------
# Function fixture: temporarily close the session source for exclusive access
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_source(
    cv_webcam_gray: OpenCVWebcamSource,
    first_webcam_index: int,
):
    """
    Close the session gray source, yield the device index, then reopen it.
    Ensures no two VideoCapture objects are open on the same device at once.
    """
    cv_webcam_gray.close()
    try:
        yield first_webcam_index
    finally:
        cv_webcam_gray.open()


# ---------------------------------------------------------------------------
# Probe / metadata
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_is_populated(self, cv_webcam_gray: OpenCVWebcamSource):
        """Size must be known after construction (probe runs in __init__)."""
        size = cv_webcam_gray.get_size()
        assert size is not None
        w, h = size
        assert w > 0 and h > 0

    def test_fps_is_positive_if_known(self, cv_webcam_gray: OpenCVWebcamSource):
        fps = cv_webcam_gray.get_fps()
        if fps is not None:
            assert fps > 0

    def test_capability_flags(self, cv_webcam_gray: OpenCVWebcamSource):
        assert cv_webcam_gray.has_timing()
        assert not cv_webcam_gray.has_num_frames()
        assert not cv_webcam_gray.is_seekable()

    def test_num_channels_grayscale(self, cv_webcam_gray: OpenCVWebcamSource):
        assert cv_webcam_gray.get_num_channels() == 1

    def test_num_channels_color(self, isolated_source: int):
        """Probe only — no open() needed, so no conflict with the session source."""
        try:
            src = OpenCVWebcamSource(isolated_source, grayscale=False)
        except Exception as exc:
            pytest.skip(f"Could not init color source: {exc}")
        assert src.get_num_channels() == 3


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_open_after_open(self, cv_webcam_gray: OpenCVWebcamSource):
        assert cv_webcam_gray.is_open()

    def test_close_stops_is_open(self, isolated_source: int):
        try:
            src = OpenCVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        assert src.is_open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self, isolated_source: int):
        try:
            src = OpenCVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        src.close()
        src.close()  # must not raise


# ---------------------------------------------------------------------------
# Read — grayscale  (uses the long-lived session source)
# ---------------------------------------------------------------------------

class TestReadGrayscale:
    def test_read_returns_video_frame(self, cv_webcam_gray: OpenCVWebcamSource):
        frame = cv_webcam_gray.read()
        assert isinstance(frame, VideoFrame)

    def test_frame_shape_is_2d(self, cv_webcam_gray: OpenCVWebcamSource):
        """Grayscale frames must have shape (H, W)."""
        frame = cv_webcam_gray.read()
        assert frame.img.ndim == 2

    def test_frame_dtype_is_uint8(self, cv_webcam_gray: OpenCVWebcamSource):
        frame = cv_webcam_gray.read()
        assert frame.img.dtype == np.uint8

    def test_frame_size_matches_probe(self, cv_webcam_gray: OpenCVWebcamSource):
        size = cv_webcam_gray.get_size()
        frame = cv_webcam_gray.read()
        h, w = frame.img.shape
        assert (w, h) == size

    def test_frame_is_not_all_zeros(self, cv_webcam_gray: OpenCVWebcamSource):
        """A real camera frame must contain at least some non-zero pixels."""
        frame = cv_webcam_gray.read()
        assert frame.img.max() > 0


# ---------------------------------------------------------------------------
# Read — color
# NOTE: These tests open their *own* source and close it immediately.
# They do NOT run concurrently with the session gray source — pytest runs
# tests serially, so the session source is idle (not mid-read) when these
# execute.  However, the session gray source still holds the device open,
# meaning we cannot open a second VideoCapture on the same device.
# We therefore run color tests with the session fixture closed then reopened
# around the test, using the `cv_webcam_color_frame` fixture below.
# ---------------------------------------------------------------------------

@pytest.fixture()
def cv_webcam_color_frame(isolated_source: int) -> np.ndarray:
    """
    Open a color source with exclusive access, grab one frame, close.
    Uses isolated_source so the session gray source is not open simultaneously.
    """
    try:
        src = OpenCVWebcamSource(isolated_source, grayscale=False)
        src.open()
        try:
            frame = src.read()
        finally:
            src.close()
    except Exception as exc:
        pytest.skip(f"Could not read color frame: {exc}")
    return frame.img


class TestReadColor:
    def test_frame_shape_is_3d(self, cv_webcam_color_frame: np.ndarray):
        assert cv_webcam_color_frame.ndim == 3
        assert cv_webcam_color_frame.shape[2] == 3

    def test_frame_dtype_is_uint8(self, cv_webcam_color_frame: np.ndarray):
        assert cv_webcam_color_frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_frame_index_starts_at_zero_after_open(self, isolated_source: int):
        """_idx must reset to 0 on each open()."""
        try:
            src = OpenCVWebcamSource(isolated_source, grayscale=True)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        try:
            f = src.read()
            assert f.frame_index == 0
        finally:
            src.close()

    def test_frame_index_increments(self, cv_webcam_gray: OpenCVWebcamSource):
        f1 = cv_webcam_gray.read()
        f2 = cv_webcam_gray.read()
        assert f2.frame_index == f1.frame_index + 1

    def test_five_consecutive_indices(self, cv_webcam_gray: OpenCVWebcamSource):
        frames = [cv_webcam_gray.read() for _ in range(5)]
        indices = [f.frame_index for f in frames]
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        assert all(d == 1 for d in diffs), f"Non-consecutive indices: {indices}"


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamp_is_non_negative(self, cv_webcam_gray: OpenCVWebcamSource):
        frame = cv_webcam_gray.read()
        assert frame.timestamp >= 0.0

    def test_timestamps_are_non_decreasing(self, cv_webcam_gray: OpenCVWebcamSource):
        frames = [cv_webcam_gray.read() for _ in range(5)]
        ts = [f.timestamp for f in frames]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
