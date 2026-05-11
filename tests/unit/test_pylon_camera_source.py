"""
Unit tests for PylonCameraSource — no hardware required.

pypylon must be installed (for module-level constants and imports), but no
physical camera is needed.  If pypylon is not installed all tests are skipped.

Fake objects
------------
FakeNode
    Duck-type for a pypylon GenICam node (Width, Height, PixelFormat, …).
    Supports GetValue() / SetValue().

FakeGrabResult
    Duck-type for pylon.GrabResult.  Controls GrabSucceeded(), Array,
    BlockID, TimeStamp, GetErrorCode(), GetErrorDescription(), Release().

FakeCamera
    Duck-type for pylon.InstantCamera.  Serves FakeGrabResult objects from a
    list via RetrieveResult(); tracks IsOpen / IsGrabbing state.

Factory pattern
---------------
_make_factory() returns a spy factory and a 'calls' list.  Each call creates
a fresh FakeCamera and appends {"serial", "camera"} to calls.

    calls[0]  — probe camera (Close()d by _probe())
    calls[1]  — live camera  (created by open())

_make_source() is a convenience wrapper.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip the whole module if pypylon is not installed.
pytest.importorskip("pypylon")

from py3r.media.types import VideoFrame
from py3r.media.video.pylon_camera_source import (
    GrabFailedError,
    GrabTimeoutError,
    PylonCameraSource,
)

W, H = 64, 48
FPS = 30.0
SERIAL = "TEST0001"


# ---------------------------------------------------------------------------
# Fake pypylon objects
# ---------------------------------------------------------------------------

class FakeNode:
    """Duck-type for a GenICam node."""

    def __init__(self, value):
        self._value = value

    def GetValue(self):
        return self._value

    def SetValue(self, value) -> None:
        self._value = value


class FakeGrabResult:
    """Duck-type for pylon.GrabResult."""

    def __init__(
        self,
        img: np.ndarray,
        *,
        grab_succeeded: bool = True,
        block_id: int | None = None,
        timestamp_ns: int | None = None,
        error_code: int = 0,
        error_desc: str = "",
    ):
        self._img = img
        self._succeeded = grab_succeeded
        self._block_id = block_id
        self._timestamp_ns = timestamp_ns
        self._error_code = error_code
        self._error_desc = error_desc
        self.released = False

    @property
    def Array(self) -> np.ndarray:
        return self._img

    @property
    def BlockID(self):
        return self._block_id

    @property
    def TimeStamp(self):
        return self._timestamp_ns

    def GrabSucceeded(self) -> bool:
        return self._succeeded

    def GetErrorCode(self) -> int:
        return self._error_code

    def GetErrorDescription(self) -> str:
        return self._error_desc

    def Release(self) -> None:
        self.released = True


class FakeCamera:
    """Duck-type for pylon.InstantCamera."""

    def __init__(
        self,
        results: list[FakeGrabResult | None],
        *,
        width: int = W,
        height: int = H,
        fps: float = FPS,
        pixel_format: str = "Mono8",
        tick_frequency: int = 1_000_000_000,
    ):
        self._results = list(results)
        self.Width = FakeNode(width)
        self.Height = FakeNode(height)
        self.AcquisitionFrameRateAbs = FakeNode(fps)
        self.PixelFormat = FakeNode(pixel_format)
        self.GevTimestampTickFrequency = FakeNode(tick_frequency)
        self.MaxNumBuffer = 0
        self._opened = True   # factory returns an already-opened camera
        self._grabbing = False
        self.start_grab_calls: list = []
        self.close_calls = 0

    def IsOpen(self) -> bool:
        return self._opened

    def IsGrabbing(self) -> bool:
        return self._grabbing

    def Open(self) -> None:
        self._opened = True

    def Close(self) -> None:
        self._opened = False
        self._grabbing = False
        self.close_calls += 1

    def StartGrabbing(self, strategy) -> None:
        self._grabbing = True
        self.start_grab_calls.append(strategy)

    def StopGrabbing(self) -> None:
        self._grabbing = False

    def GetNodeMap(self):
        return MagicMock()

    def RetrieveResult(self, timeout_ms: int, handling) -> FakeGrabResult | None:
        if not self._results:
            return None
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _gray_img(val: int = 128) -> np.ndarray:
    return np.full((H, W), val, dtype=np.uint8)


def _make_result(
    val: int = 128,
    *,
    block_id: int | None = None,
    timestamp_ns: int | None = None,
) -> FakeGrabResult:
    return FakeGrabResult(
        _gray_img(val),
        grab_succeeded=True,
        block_id=block_id,
        timestamp_ns=timestamp_ns,
    )


def _make_factory(
    results: list[FakeGrabResult | None],
    *,
    width: int = W,
    height: int = H,
    fps: float = FPS,
    pixel_format: str = "Mono8",
    tick_frequency: int = 1_000_000_000,
):
    """Return (factory, calls).  Each factory call creates a fresh FakeCamera."""
    calls: list[dict] = []

    def factory(serial: str):
        cam = FakeCamera(list(results), width=width, height=height,
                         fps=fps, pixel_format=pixel_format,
                         tick_frequency=tick_frequency)
        calls.append({"serial": serial, "camera": cam})
        return cam

    return factory, calls


def _make_source(
    results: list[FakeGrabResult | None],
    *,
    width: int = W,
    height: int = H,
    fps: float = FPS,
    pixel_format: str = "Mono8",
    tick_frequency: int = 1_000_000_000,
):
    """Convenience wrapper.  Returns (src, calls); calls[0]=probe, calls[1]=live."""
    factory, calls = _make_factory(results, width=width, height=height,
                                   fps=fps, pixel_format=pixel_format,
                                   tick_frequency=tick_frequency)
    src = PylonCameraSource(SERIAL, camera_factory=factory)
    return src, calls


def _live(calls: list[dict]) -> FakeCamera:
    return calls[1]["camera"]


# ---------------------------------------------------------------------------
# TestCameraFactoryArgs
# ---------------------------------------------------------------------------

class TestCameraFactoryArgs:
    def test_serial_passed_to_factory(self):
        factory, calls = _make_factory([])
        PylonCameraSource("SN123", camera_factory=factory)
        assert calls[0]["serial"] == "SN123"

    def test_factory_called_once_for_probe(self):
        factory, calls = _make_factory([])
        PylonCameraSource(SERIAL, camera_factory=factory)
        assert len(calls) == 1

    def test_factory_called_again_for_open(self):
        src, calls = _make_source([])
        src.open()
        assert len(calls) == 2

    def test_probe_camera_is_closed_after_probe(self):
        src, calls = _make_source([])
        assert calls[0]["camera"].close_calls >= 1

    def test_probe_camera_not_open_after_probe(self):
        src, calls = _make_source([])
        assert not calls[0]["camera"].IsOpen()


# ---------------------------------------------------------------------------
# TestProbe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_from_probe(self):
        src, _ = _make_source([], width=320, height=240)
        assert src.get_size() == (320, 240)

    def test_fps_from_probe(self):
        src, _ = _make_source([], fps=60.0)
        assert src.get_fps() == 60.0

    def test_has_size_after_probe(self):
        src, _ = _make_source([])
        assert src.has_size()

    def test_has_fps_after_probe(self):
        src, _ = _make_source([])
        assert src.has_fps()

    def test_grayscale_when_mono8(self):
        src, _ = _make_source([], pixel_format="Mono8")
        assert src.get_num_channels() == 1

    def test_color_when_not_mono8(self):
        src, _ = _make_source([], pixel_format="BayerRG8")
        assert src.get_num_channels() == 3

    def test_probe_soft_fails_when_factory_raises(self):
        """Construction must not raise even when the factory fails."""
        def bad_factory(serial: str):
            raise RuntimeError("no camera")

        src = PylonCameraSource(SERIAL, camera_factory=bad_factory)
        assert src.get_size() is None
        assert not src.has_size()


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_not_open_before_open(self):
        src, _ = _make_source([])
        assert not src.is_open()

    def test_is_open_after_open(self):
        src, _ = _make_source([])
        src.open()
        assert src.is_open()

    def test_not_open_after_close(self):
        src, _ = _make_source([])
        src.open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self):
        src, _ = _make_source([])
        src.open()
        src.close()
        src.close()

    def test_open_starts_grabbing(self):
        src, calls = _make_source([])
        src.open()
        assert _live(calls).IsGrabbing()

    def test_open_calls_start_grabbing_once(self):
        src, calls = _make_source([])
        src.open()
        assert len(_live(calls).start_grab_calls) == 1

    def test_close_stops_grabbing(self):
        src, calls = _make_source([])
        src.open()
        src.close()
        # The live camera is closed; IsGrabbing checks internal state.
        assert not _live(calls).IsGrabbing()

    def test_capability_flags(self):
        src, _ = _make_source([])
        assert src.has_timing()
        assert not src.has_num_frames()
        assert not src.is_seekable()


# ---------------------------------------------------------------------------
# TestRead
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_returns_video_frame(self):
        src, _ = _make_source([_make_result()])
        src.open()
        frame = src.read()
        assert isinstance(frame, VideoFrame)

    def test_frame_shape_matches_camera(self):
        src, _ = _make_source([_make_result()])
        src.open()
        frame = src.read()
        assert frame.img.shape == (H, W)

    def test_frame_pixel_values(self):
        src, _ = _make_source([_make_result(200)])
        src.open()
        assert src.read().img[0, 0] == 200

    def test_result_released_after_read(self):
        result = _make_result()
        src, _ = _make_source([result])
        src.open()
        src.read()
        assert result.released

    def test_frame_index_resets_on_open(self):
        factory, _ = _make_factory([_make_result(), _make_result()])
        src = PylonCameraSource(SERIAL, camera_factory=factory)
        src.open()
        src.read()
        assert src._idx == 1
        src.close()
        src.open()
        assert src._idx == 0


# ---------------------------------------------------------------------------
# TestFrameIndex
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_uses_block_id_when_present(self):
        """BlockID from the SDK must be used as the frame_index."""
        result = _make_result(block_id=42)
        src, _ = _make_source([result])
        src.open()
        assert src.read().frame_index == 42

    def test_falls_back_to_internal_counter_when_no_block_id(self):
        result = _make_result(block_id=None)
        src, _ = _make_source([result])
        src.open()
        assert src.read().frame_index == 0

    def test_internal_counter_increments_each_read(self):
        results = [_make_result(block_id=None), _make_result(block_id=None)]
        src, _ = _make_source(results)
        src.open()
        src.read()
        assert src._idx == 1
        src.read()
        assert src._idx == 2

    def test_block_id_gap_is_preserved(self):
        """If camera skips block IDs (dropped frames), frame_index must reflect that."""
        r1 = _make_result(block_id=10)
        r2 = _make_result(block_id=15)   # gap of 4
        src, _ = _make_source([r1, r2])
        src.open()
        f1 = src.read()
        f2 = src.read()
        assert f2.frame_index - f1.frame_index == 5


# ---------------------------------------------------------------------------
# TestTimestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamp_from_result_at_1ghz(self):
        """Default tick frequency is 1 GHz (1e9 ticks/s) → ticks / 1e9 = seconds."""
        result = _make_result(timestamp_ns=2_000_000_000)  # 2 billion ticks @ 1 GHz = 2 s
        src, _ = _make_source([result])
        src.open()
        assert src.read().timestamp == pytest.approx(2.0)

    def test_timestamp_uses_tick_frequency_from_probe(self):
        """When GevTimestampTickFrequency is e.g. 125 MHz, ticks/125e6 = seconds."""
        tick_freq = 125_000_000  # 125 MHz — common Basler GigE value
        ticks = 250_000_000      # 250 million ticks → 2 seconds
        result = _make_result(timestamp_ns=ticks)
        src, _ = _make_source([result], tick_frequency=tick_freq)
        src.open()
        assert src.read().timestamp == pytest.approx(2.0)

    def test_timestamp_zero_falls_back_to_perf_counter(self):
        """TimeStamp == 0 (falsy) must fall back to time.perf_counter()."""
        result = FakeGrabResult(_gray_img(), timestamp_ns=0)
        src, _ = _make_source([result])
        src.open()
        before = time.perf_counter()
        ts = src.read().timestamp
        after = time.perf_counter()
        assert before <= ts <= after

    def test_timestamp_none_falls_back_to_perf_counter(self):
        result = FakeGrabResult(_gray_img(), timestamp_ns=None)
        src, _ = _make_source([result])
        src.open()
        before = time.perf_counter()
        ts = src.read().timestamp
        after = time.perf_counter()
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# TestGrabFailed
# ---------------------------------------------------------------------------

class TestGrabFailed:
    def test_grab_failed_raises_grab_failed_error(self):
        result = FakeGrabResult(
            _gray_img(),
            grab_succeeded=False,
            error_code=0xE1000014,
            error_desc="Buffer incomplete",
        )
        src, _ = _make_source([result])
        src.open()
        with pytest.raises(GrabFailedError):
            src.read()

    def test_grab_failed_message_contains_error_code(self):
        result = FakeGrabResult(_gray_img(), grab_succeeded=False,
                                error_code=0xDEAD, error_desc="")
        src, _ = _make_source([result])
        src.open()
        with pytest.raises(GrabFailedError, match="0x0000dead"):
            src.read()

    def test_grab_failed_releases_result(self):
        result = FakeGrabResult(_gray_img(), grab_succeeded=False)
        src, _ = _make_source([result])
        src.open()
        with pytest.raises(GrabFailedError):
            src.read()
        assert result.released


# ---------------------------------------------------------------------------
# TestGrabTimeout
# ---------------------------------------------------------------------------

class TestGrabTimeout:
    def test_retrieve_none_raises_grab_timeout(self):
        """RetrieveResult returning None means no frame in time → GrabTimeoutError."""
        src, _ = _make_source([None])
        src.open()
        with pytest.raises(GrabTimeoutError):
            src.read()

    def test_grab_timeout_message_contains_duration(self):
        src, _ = _make_source([None])
        src.open()
        with pytest.raises(GrabTimeoutError, match="0.250"):
            src.read(timeout=0.25)



# ---------------------------------------------------------------------------
# TestNotOpen
# ---------------------------------------------------------------------------

class TestNotOpen:
    def test_read_before_open_raises_runtime_error(self):
        src, _ = _make_source([])
        with pytest.raises(RuntimeError, match="not open"):
            src.read()

    def test_read_after_close_raises_runtime_error(self):
        src, _ = _make_source([])
        src.open()
        src.close()
        with pytest.raises(RuntimeError):
            src.read()




