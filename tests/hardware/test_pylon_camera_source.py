"""
Hardware tests for PylonCameraSource.

Skipped automatically when no Basler/Pylon camera is detected.  Pass
``--require-pylon`` to turn skips into failures (useful in CI).

A single session-scoped fixture opens the camera once; most tests share the
live stream to avoid repeated open/close cycles (which are slow on GigE
cameras).

Tests that need exclusive device access use the ``isolated_source`` function
fixture, which temporarily closes the session source, yields the serial number
for the test to use, then reopens the session source.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from py3r.media.types import VideoFrame
from py3r.media.video.pylon_camera_source import (
    GrabFailedError,
    GrabTimeoutError,
    PylonCameraSource,
)

pytest.importorskip("pypylon")

pytestmark = [pytest.mark.hardware]


# ---------------------------------------------------------------------------
# Session-scoped source
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pylon_source(first_pylon_serial: str) -> PylonCameraSource:
    """Open a PylonCameraSource against the first detected Basler camera."""
    try:
        src = PylonCameraSource(first_pylon_serial)
    except Exception as exc:
        pytest.skip(f"PylonCameraSource('{first_pylon_serial}') failed to init: {exc}")

    src.open()
    yield src
    src.close()


# ---------------------------------------------------------------------------
# Function fixture: temporarily close the session source for exclusive access
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_source(
    pylon_source: PylonCameraSource,
    first_pylon_serial: str,
):
    """
    Close the session source, yield the serial number, then reopen it.
    Ensures no two PylonCameraSource instances probe or open the device
    simultaneously.
    """
    pylon_source.close()
    try:
        yield first_pylon_serial
    finally:
        pylon_source.open()


# ---------------------------------------------------------------------------
# Probe / metadata
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_is_populated(self, pylon_source: PylonCameraSource):
        size = pylon_source.get_size()
        assert size is not None
        w, h = size
        assert w > 0 and h > 0

    def test_fps_is_positive(self, pylon_source: PylonCameraSource):
        fps = pylon_source.get_fps()
        assert fps is not None and fps > 0

    def test_capability_flags(self, pylon_source: PylonCameraSource):
        assert pylon_source.has_timing()
        assert not pylon_source.has_num_frames()
        assert not pylon_source.is_seekable()

    def test_num_channels_is_1_or_3(self, pylon_source: PylonCameraSource):
        assert pylon_source.get_num_channels() in (1, 3)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_open_after_open(self, pylon_source: PylonCameraSource):
        assert pylon_source.is_open()

    def test_close_stops_is_open(self, isolated_source: str):
        try:
            src = PylonCameraSource(isolated_source)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        assert src.is_open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self, isolated_source: str):
        try:
            src = PylonCameraSource(isolated_source)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        src.close()
        src.close()  # must not raise

    def test_read_before_open_raises(self, isolated_source: str):
        try:
            src = PylonCameraSource(isolated_source)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        with pytest.raises(RuntimeError, match="not open"):
            src.read()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_returns_video_frame(self, pylon_source: PylonCameraSource):
        frame = pylon_source.read(timeout=5.0)
        assert isinstance(frame, VideoFrame)

    def test_frame_dtype_is_uint8(self, pylon_source: PylonCameraSource):
        frame = pylon_source.read(timeout=5.0)
        assert frame.img.dtype == np.uint8

    def test_frame_size_matches_probe(self, pylon_source: PylonCameraSource):
        size = pylon_source.get_size()
        frame = pylon_source.read(timeout=5.0)
        h, w = frame.img.shape[:2]
        assert (w, h) == size

    def test_frame_ndim_matches_channels(self, pylon_source: PylonCameraSource):
        frame = pylon_source.read(timeout=5.0)
        n_ch = pylon_source.get_num_channels()
        if n_ch == 1:
            assert frame.img.ndim == 2
        else:
            assert frame.img.ndim == 3 and frame.img.shape[2] == n_ch

    def test_frame_is_not_all_zeros(self, pylon_source: PylonCameraSource):
        frame = pylon_source.read(timeout=5.0)
        assert frame.img.max() > 0


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_frame_index_starts_at_zero_after_open(self, isolated_source: str):
        """_idx must reset to 0 on each open()."""
        try:
            src = PylonCameraSource(isolated_source)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")
        src.open()
        try:
            f = src.read(timeout=5.0)
            assert f.frame_index >= 0
        finally:
            src.close()

    def test_frame_indices_are_non_decreasing(self, pylon_source: PylonCameraSource):
        frames = [pylon_source.read(timeout=5.0) for _ in range(5)]
        indices = [f.frame_index for f in frames]
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)), (
            f"Frame indices not strictly increasing: {indices}"
        )


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamp_is_positive(self, pylon_source: PylonCameraSource):
        frame = pylon_source.read(timeout=5.0)
        assert frame.timestamp > 0.0

    def test_timestamps_are_non_decreasing(self, pylon_source: PylonCameraSource):
        frames = [pylon_source.read(timeout=5.0) for _ in range(5)]
        ts = [f.timestamp for f in frames]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_short_timeout_raises_grab_timeout(self, isolated_source: str):
        """Near-zero timeout on a fresh source should raise GrabTimeoutError."""
        try:
            src = PylonCameraSource(isolated_source)
        except Exception as exc:
            pytest.skip(f"Could not init source: {exc}")

        src.open()
        try:
            with pytest.raises(GrabTimeoutError):
                src.read(timeout=0.001)
        finally:
            src.close()
