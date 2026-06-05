"""
Unit tests for PyAVWebcamSource — no hardware required.

All calls to ``av.open`` are patched out.  Reader-loop behaviour (frame
delivery, EOF, errors, timeout, queue saturation) is covered separately in
``test_pyav_container_reader.py``.

Test doubles
------------
FakeContainer
    Minimal duck-type for an ``av.InputContainer``.  Provides a ``streams``
    list with one pre-configured MagicMock video stream (W×H @ FPS) and a
    ``close()`` method.  ``decode()`` terminates immediately (no frames), so
    probing never blocks.

Patching strategy
-----------------
``av.open`` is patched with a ``side_effect`` callable that returns a fresh
``FakeContainer`` on every call.  This means the probe container (created
during ``__init__``) and the live container (created during ``open()``) are
independent objects.  Tests that need to inspect call arguments use
``mock_open.call_args`` or ``mock_open.call_args_list``.
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from py3r.media.video.pyav_webcam_source import PyAVWebcamSource

W, H = 64, 48
FPS = 30.0
DEVICE = "MockCamera"
EXPECTED_FORMAT = "dshow" if sys.platform == "win32" else (
    "avfoundation" if sys.platform == "darwin" else "v4l2"
)


# ---------------------------------------------------------------------------
# Fake container
# ---------------------------------------------------------------------------

class FakeContainer:
    """
    Minimal av.InputContainer duck-type.

    decode() either blocks until close() or terminates immediately (no frames).
    streams[0] is a MagicMock configured with W, H, FPS so that
    _update_size_and_fps_from_stream extracts real metadata without touching av.
    """

    def __init__(self, *, block_until_closed: bool = False):
        self._block_until_closed = block_until_closed
        self._closed = threading.Event()
        self.all_frames_yielded = threading.Event()

        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.width = W
        mock_stream.codec_context.height = H
        mock_stream.average_rate = FPS

        mock_stream2 = MagicMock()
        mock_stream2.type = "video"
        mock_stream2.codec_context.width = W
        mock_stream2.codec_context.height = H
        mock_stream2.average_rate = FPS

        self.streams = [mock_stream, mock_stream2]

    def decode(self, stream):  # noqa: ARG002
        if self._block_until_closed:
            self._closed.wait()
            return
        self.all_frames_yielded.set()
        return
        yield  # make this a generator

    def close(self) -> None:
        self._closed.set()


def _container_factory(*, block_until_closed: bool = False):
    """Return a side_effect callable that creates a fresh FakeContainer per call."""
    def factory(**_kw):
        return FakeContainer(block_until_closed=block_until_closed)
    return factory


# ---------------------------------------------------------------------------
# av.open arguments (tests _open_container call site)
# ---------------------------------------------------------------------------

class TestOpenArgs:
    """Verify that _open_container passes the right arguments to av.open."""

    def _probe_kwargs(self, **src_kwargs) -> dict:
        with patch("av.open", side_effect=_container_factory()) as mock_open:
            PyAVWebcamSource(DEVICE, **src_kwargs)
        return mock_open.call_args.kwargs

    def test_file_is_device_name(self):
        kw = self._probe_kwargs()
        if kw["format"] == "dshow":
            assert kw["file"] == f"video={DEVICE}"
        else:
            assert kw["file"] == DEVICE

    def test_format_is_platform_default(self):
        assert self._probe_kwargs()["format"] == EXPECTED_FORMAT

    def test_explicit_format_is_forwarded(self):
        assert self._probe_kwargs(input_format="v4l2")["format"] == "v4l2"

    def test_mode_is_read(self):
        assert self._probe_kwargs()["mode"] == "r"

    def test_options_video_size(self):
        opts = self._probe_kwargs(width=1280, height=720)["options"]
        assert opts["video_size"] == "1280x720"

    def test_options_framerate(self):
        opts = self._probe_kwargs(fps=60.0)["options"]
        assert opts["framerate"] == "60.0"

    def test_options_rtbufsize_on_dshow(self):
        with patch("av.open", side_effect=_container_factory()) as mock_open:
            PyAVWebcamSource(DEVICE, input_format="dshow")
        assert mock_open.call_args.kwargs["options"].get("rtbufsize") == "500M"

    def test_no_rtbufsize_on_v4l2(self):
        with patch("av.open", side_effect=_container_factory()) as mock_open:
            PyAVWebcamSource(DEVICE, input_format="v4l2")
        assert "rtbufsize" not in mock_open.call_args.kwargs["options"]

    def test_no_size_options_when_not_specified(self):
        assert "video_size" not in self._probe_kwargs()["options"]

    def test_no_framerate_option_when_not_specified(self):
        assert "framerate" not in self._probe_kwargs()["options"]

    def test_probe_and_open_receive_identical_args(self):
        """Both probe and live session pass identical arguments to av.open."""
        seen: list[dict] = []

        def factory(**kw):
            seen.append(dict(kw))
            return FakeContainer(block_until_closed=True)

        with patch("av.open", side_effect=factory):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            src.close()

        assert len(seen) == 2
        probe_kw, live_kw = seen
        for key in ("file", "format", "mode", "options"):
            assert probe_kw[key] == live_kw[key], f"Mismatch on {key!r}"

    def test_raises_without_device_name(self):
        with pytest.raises(TypeError):
            PyAVWebcamSource()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Probe — metadata extraction
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size_from_stream_metadata(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
        assert src.get_size() == (W, H)

    def test_fps_from_stream_metadata(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
        assert src.get_fps() == pytest.approx(FPS)

    def test_given_size_skips_probe_decode(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE, width=320, height=240)
        assert src.get_size() == (320, 240)

    def test_probe_failure_leaves_size_none(self):
        with patch("av.open", side_effect=RuntimeError("no device")):
            src = PyAVWebcamSource(DEVICE)
        assert src.get_size() is None

    def test_has_size_after_probe(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
        assert src.has_size()

    def test_has_fps_after_probe(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
        assert src.has_fps()


# ---------------------------------------------------------------------------
# Stream index
# ---------------------------------------------------------------------------

class TestStreamIndex:
    def _container_with_streams(self, *types):
        """Build a mock container whose streams have the given type strings."""
        streams = []
        for t in types:
            s = MagicMock()
            s.type = t
            streams.append(s)
        container = MagicMock()
        container.streams = streams
        return container, streams

    def test_none_selects_first_video_stream(self):
        container, streams = self._container_with_streams("video", "video")
        assert PyAVWebcamSource._select_video_stream(container) is streams[0]

    def test_none_skips_non_video_to_find_first_video(self):
        container, streams = self._container_with_streams("audio", "video")
        assert PyAVWebcamSource._select_video_stream(container) is streams[1]

    def test_integer_indexes_all_streams(self):
        container, streams = self._container_with_streams("audio", "video")
        assert PyAVWebcamSource._select_video_stream(container, stream_index=1) is streams[1]

    def test_integer_selects_second_video_stream_by_absolute_index(self):
        container, streams = self._container_with_streams("video", "video")
        assert PyAVWebcamSource._select_video_stream(container, stream_index=1) is streams[1]

    def test_integer_non_video_raises(self):
        container, _ = self._container_with_streams("audio", "video")
        with pytest.raises(RuntimeError, match="not a video stream"):
            PyAVWebcamSource._select_video_stream(container, stream_index=0)

    def test_integer_out_of_range_raises(self):
        container, _ = self._container_with_streams("video")
        with pytest.raises(RuntimeError, match="out of range"):
            PyAVWebcamSource._select_video_stream(container, stream_index=5)

    def test_none_no_video_stream_raises(self):
        container, _ = self._container_with_streams("audio", "data")
        with pytest.raises(RuntimeError, match="No video stream"):
            PyAVWebcamSource._select_video_stream(container)

    def test_empty_streams_none_raises(self):
        container = MagicMock()
        container.streams = []
        with pytest.raises(RuntimeError, match="No video stream"):
            PyAVWebcamSource._select_video_stream(container)

    def test_stream_index_forwarded_to_select(self):
        """stream_index=1 passed to PyAVWebcamSource reaches _select_video_stream."""
        seen_indices: list = []
        original = PyAVWebcamSource._select_video_stream

        def spy(container, stream_index=None):
            seen_indices.append(stream_index)
            return original(container, stream_index)

        with patch("av.open", side_effect=_container_factory()):
            with patch.object(PyAVWebcamSource, "_select_video_stream",
                              staticmethod(spy)):
                PyAVWebcamSource(DEVICE, stream_index=1)

        assert 1 in seen_indices


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_open_before_open(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
            assert not src.is_open()

    def test_is_open_after_open(self):
        with patch("av.open", side_effect=_container_factory(block_until_closed=True)):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            assert src.is_open()
            src.close()

    def test_is_closed_after_close(self):
        with patch("av.open", side_effect=_container_factory(block_until_closed=True)):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            src.close()
            assert not src.is_open()

    def test_double_close_is_safe(self):
        with patch("av.open", side_effect=_container_factory(block_until_closed=True)):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            src.close()
            src.close()

    def test_open_is_idempotent(self):
        with patch("av.open", side_effect=_container_factory(block_until_closed=True)):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            reader_before = src._container_reader
            src.open()
            assert src._container_reader is reader_before
            src.close()

    def test_frame_index_resets_on_reopen(self):
        """After close() + open(), the reader restarts at frame index 0."""
        import numpy as np
        from tests.unit.test_pyav_container_reader import FakeAVFrame
        from tests.unit.test_pyav_container_reader import FakeContainer as FC

        frames = [
            FakeAVFrame(np.full((H, W), i * 80, dtype=np.uint8), time=i / FPS)
            for i in range(3)
        ]
        containers: list = []

        def factory(**_kw):
            c = FC(frames[:])
            containers.append(c)
            return c

        with patch("av.open", side_effect=factory):
            src = PyAVWebcamSource(DEVICE, width=W, height=H, fps=FPS)
            src.open()
            assert containers[-1].all_frames_yielded.wait(timeout=3.0)
            for _ in range(3):
                src.read(timeout=2.0)
            src.close()

            src.open()
            assert containers[-1].all_frames_yielded.wait(timeout=3.0)
            frame = src.read(timeout=2.0)
            src.close()

        assert frame.frame_index == 0


# ---------------------------------------------------------------------------
# Not open
# ---------------------------------------------------------------------------

class TestNotOpen:
    def test_read_raises_when_not_open(self):
        with patch("av.open", side_effect=_container_factory()):
            src = PyAVWebcamSource(DEVICE)
        with pytest.raises(RuntimeError, match="not open"):
            src.read()

