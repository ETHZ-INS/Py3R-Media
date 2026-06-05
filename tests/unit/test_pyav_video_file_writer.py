"""
Unit tests for PyAVVideoFileWriter.

Strategy
--------
Write frames to a ``tmp_path`` file, then read them back with ``av.open``
and compare.  No mocking is needed — the output file is the natural injection
point for a file writer, exactly as a real input file is for the file source
tests.

Round-trip fidelity
-------------------
* **Lossless** (libx264/libx264rgb with crf=0): pixel values are preserved
  exactly — tests assert ``np.array_equal``.
* **Lossy** (libx264 + yuv420p): solid-fill images survive the YUV round-trip
  exactly for the luma channel, so grey-value tests still use exact comparison.
  Color lossy tests only check frame count and shape.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from py3r.media.types import VideoFrame
from py3r.media.video.pyav_video_file_writer import (
    EncoderConfig,
    PyAVVideoFileWriter,
)

W, H = 64, 48
FPS = 30.0
N = 10  # frames per test clip


# ---------------------------------------------------------------------------
# Frame-generation helpers
# ---------------------------------------------------------------------------

def _gray(n: int = N, start: int = 0) -> list[np.ndarray]:
    """Solid-fill grayscale frames with linearly increasing brightness."""
    return [np.full((H, W), (start + i * 8) % 256, dtype=np.uint8) for i in range(n)]


def _color(n: int = N, start: int = 0) -> list[np.ndarray]:
    """Solid-fill BGR color frames (R=G=B for reliable YUV round-trip)."""
    return [np.full((H, W, 3), (start + i * 8) % 256, dtype=np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------
# Read-back helper
# ---------------------------------------------------------------------------

def _read_frames(path: Path, fmt: str) -> list[np.ndarray]:
    """Decode all frames from *path* and return them as ndarrays in *fmt*,
    sorted by PTS so B-frame reordering doesn't affect comparisons."""
    frames: list[tuple[int | None, np.ndarray]] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append((frame.pts, frame.to_ndarray(format=fmt)))
    frames.sort(key=lambda x: (x[0] is None, x[0]))
    return [arr for _, arr in frames]


def _make_writer(path: Path, *, grayscale: bool = True,
                 encoder_config: EncoderConfig | None = None,
                 **kw) -> PyAVVideoFileWriter:
    if encoder_config is None:
        encoder_config = _lossy_gray_config() if grayscale else _lossy_color_config()
    return PyAVVideoFileWriter(path, size=(W, H), fps=FPS,
                               encoder_config=encoder_config,
                               input_pix_fmt="gray" if grayscale else "bgr24",
                               **kw)


def _lossy_gray_config(crf: str = "28", preset: str = "ultrafast",
                       extra: dict | None = None) -> EncoderConfig:
    opts = {"crf": crf, "preset": preset}
    if extra:
        opts.update(extra)
    return EncoderConfig(codec="libx264", pix_fmt="yuv420p", options=opts)


def _lossy_color_config(crf: str = "28", preset: str = "ultrafast") -> EncoderConfig:
    return EncoderConfig(codec="libx264", pix_fmt="yuv420p",
                         options={"crf": crf, "preset": preset})


def _lossless_gray_config(extra: dict | None = None) -> EncoderConfig:
    opts = {"crf": "0", "preset": "veryslow"}
    if extra:
        opts.update(extra)
    return EncoderConfig(
        codec="libx264", pix_fmt="gray", options=opts,
        codec_context_attrs={"color_range": 2},
    )


def _lossless_color_config() -> EncoderConfig:
    return EncoderConfig(codec="libx264rgb", pix_fmt="bgr24",
                         options={"crf": "0", "preset": "veryslow"})


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_not_open_before_open(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        assert not w.is_open

    def test_is_open_after_open(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        w.open()
        assert w.is_open
        w.close()

    def test_not_open_after_close(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        w.open()
        w.close()
        assert not w.is_open

    def test_double_open_is_no_op(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        w.open()
        container_id = id(w._container)
        w.open()  # second call must not replace the container
        assert id(w._container) == container_id
        w.close()

    def test_double_close_is_safe(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        w.open()
        w.close()
        w.close()  # must not raise

    def test_context_manager_opens_and_closes(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        with w:
            assert w.is_open
        assert not w.is_open

    def test_context_manager_closes_on_exception(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        with pytest.raises(ZeroDivisionError):
            with w:
                raise ZeroDivisionError
        assert not w.is_open

    def test_file_is_created_after_close(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            w.write(_gray(1)[0])  # at least one frame is required
        assert path.exists()


# ---------------------------------------------------------------------------
# TestWrite
# ---------------------------------------------------------------------------

class TestWrite:
    def test_frame_count_increments(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        w.open()
        for _ in range(5):
            w.write(_gray(1)[0])
        assert w._frame_count == 5
        w.close()

    def test_correct_number_of_frames_in_file(self, tmp_path):
        path = tmp_path / "out.mp4"
        frames = _gray(N)
        with _make_writer(path) as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "gray")
        assert len(decoded) == N

    def test_gray_frame_shape_in_file(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            w.write(_gray(1)[0])
        decoded = _read_frames(path, "gray")
        assert decoded[0].shape == (H, W)

    def test_color_frame_shape_in_file(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path, grayscale=False) as w:
            w.write(_color(1)[0])
        decoded = _read_frames(path, "bgr24")
        assert decoded[0].shape == (H, W, 3)

    def test_accepts_has_image_object(self, tmp_path):
        """write() must accept any object with a .img attribute (HasImage protocol)."""
        path = tmp_path / "out.mp4"
        img = _gray(1)[0]
        vf = VideoFrame(img, 0, 0.0)  # VideoFrame has .img
        with _make_writer(path) as w:
            w.write(vf)  # must not raise
        decoded = _read_frames(path, "gray")
        assert len(decoded) == 1

    def test_accepts_non_contiguous_array(self, tmp_path):
        """Sliced (non-contiguous) arrays must be handled without error."""
        path = tmp_path / "out.mp4"
        base = np.full((H * 2, W), 100, dtype=np.uint8)
        non_contig = base[::2]  # every other row → non-contiguous
        with _make_writer(path) as w:
            w.write(non_contig)


# ---------------------------------------------------------------------------
# TestRoundTrip — pixel-level fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_lossless_gray_exact(self, tmp_path):
        """All pixel values must survive the lossless encode/decode cycle exactly.
        A constant fill is used so B-frame reordering doesn't affect comparison."""
        path = tmp_path / "out.mkv"
        frame = np.full((H, W), 150, dtype=np.uint8)
        with _make_writer(path, encoder_config=_lossless_gray_config()) as w:
            for _ in range(5):
                w.write(frame.copy())
        decoded = _read_frames(path, "gray")
        assert len(decoded) == 5
        for dec in decoded:
            np.testing.assert_array_equal(dec, frame)

    def test_lossless_gray_different_values_exact(self, tmp_path):
        """Different pixel values must survive lossless encode/decode exactly.
        B-frames are explicitly disabled so PTS-sorted order matches write order."""
        path = tmp_path / "out.mkv"
        frames = _gray(5, start=50)  # values 50, 58, 66, 74, 82 — well within valid range
        cfg = _lossless_gray_config(extra={"bf": "0"})
        with _make_writer(path, encoder_config=cfg) as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "gray")
        assert len(decoded) == len(frames)
        for orig, dec in zip(frames, decoded):
            np.testing.assert_array_equal(orig, dec)

    def test_lossless_color_exact(self, tmp_path):
        path = tmp_path / "out.mkv"
        frames = _color(5)
        with _make_writer(path, grayscale=False,
                          encoder_config=_lossless_color_config()) as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "bgr24")
        for orig, dec in zip(frames, decoded):
            np.testing.assert_array_equal(orig, dec)

    def test_lossy_gray_solid_fill_survives(self, tmp_path):
        """Solid-fill luma values must survive yuv420p encode/decode within ±1 LSB
        (YUV limited-range rounding may shift values by 1 count)."""
        path = tmp_path / "out.mp4"
        frames = _gray(5)
        with _make_writer(path) as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "gray")
        assert len(decoded) == len(frames)
        for orig, dec in zip(frames, decoded):
            np.testing.assert_allclose(dec.astype(int), orig.astype(int), atol=2)

    def test_hw_size_preserved(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            w.write(_gray(1)[0])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.width == W
            assert stream.height == H


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_write_before_open_raises_runtime_error(self, tmp_path):
        w = _make_writer(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError):
            w.write(_gray(1)[0])

    def test_wrong_dtype_raises_value_error(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            bad = np.full((H, W), 0.5, dtype=np.float32)
            with pytest.raises(ValueError, match="uint8"):
                w.write(bad)

    def test_gray_mode_rejects_color_array(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path, grayscale=True) as w:
            with pytest.raises(ValueError):
                w.write(_color(1)[0])  # (H, W, 3) into gray writer

    def test_color_mode_rejects_gray_array(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path, grayscale=False) as w:
            with pytest.raises(ValueError):
                w.write(_gray(1)[0])  # (H, W) into color writer

    def test_wrong_height_raises_value_error(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            wrong = np.full((H + 1, W), 0, dtype=np.uint8)
            with pytest.raises(ValueError):
                w.write(wrong)

    def test_wrong_width_raises_value_error(self, tmp_path):
        path = tmp_path / "out.mp4"
        with _make_writer(path) as w:
            wrong = np.full((H, W + 1), 0, dtype=np.uint8)
            with pytest.raises(ValueError):
                w.write(wrong)

    def test_gray_hw1_shape_is_accepted(self, tmp_path):
        """(H, W, 1) grayscale must be silently squeezed to (H, W)."""
        path = tmp_path / "out.mp4"
        with _make_writer(path, grayscale=True) as w:
            frame = np.full((H, W, 1), 128, dtype=np.uint8)
            w.write(frame)  # must not raise


# ---------------------------------------------------------------------------
# TestEncoderConfig — different encoder setups produce valid files
# ---------------------------------------------------------------------------

class TestEncoderConfig:
    @pytest.mark.parametrize("crf,preset", [
        ("36", "ultrafast"),
        ("32", "ultrafast"),
        ("28", "veryfast"),
        ("22", "medium"),
        ("18", "fast"),
    ])
    def test_lossy_gray_config_produces_valid_file(self, tmp_path, crf, preset):
        path = tmp_path / "out.mp4"
        with _make_writer(path, encoder_config=_lossy_gray_config(crf=crf, preset=preset)) as w:
            w.write(_gray(1)[0])
        assert len(_read_frames(path, "gray")) == 1

    def test_lossless_gray_produces_valid_file(self, tmp_path):
        path = tmp_path / "out.mkv"
        with _make_writer(path, encoder_config=_lossless_gray_config()) as w:
            w.write(_gray(1)[0])
        assert len(_read_frames(path, "gray")) == 1

    def test_lossless_color_produces_valid_file(self, tmp_path):
        path = tmp_path / "out.mkv"
        with _make_writer(path, grayscale=False,
                          encoder_config=_lossless_color_config()) as w:
            w.write(_color(1)[0])
        assert len(_read_frames(path, "bgr24")) == 1

    def test_codec_context_attrs_applied(self, tmp_path):
        """codec_context_attrs must be set on the stream without error."""
        path = tmp_path / "out.mp4"
        cfg = EncoderConfig(
            codec="libx264", pix_fmt="gray",
            options={"crf": "28", "preset": "ultrafast"},
            codec_context_attrs={"thread_count": 1},
        )
        with _make_writer(path, encoder_config=cfg) as w:
            w.write(_gray(1)[0])
        assert len(_read_frames(path, "gray")) == 1


# ---------------------------------------------------------------------------
# TestPyAVVideoFileWriter — core class with explicit EncoderConfig
# ---------------------------------------------------------------------------

class TestPyAVVideoFileWriter:
    def _make(self, path, *, input_pix_fmt: str = "gray") -> PyAVVideoFileWriter:
        cfg = EncoderConfig(
            codec="libx264",
            pix_fmt="yuv420p",
            options={"crf": "28", "preset": "ultrafast"},
        )
        return PyAVVideoFileWriter(path, size=(W, H), fps=FPS, encoder_config=cfg,
                                   input_pix_fmt=input_pix_fmt)

    def test_not_open_before_open(self, tmp_path):
        assert not self._make(tmp_path / "out.mp4").is_open

    def test_is_open_after_open(self, tmp_path):
        w = self._make(tmp_path / "out.mp4")
        w.open()
        assert w.is_open
        w.close()

    def test_write_raises_when_not_open(self, tmp_path):
        w = self._make(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError):
            w.write(_gray(1)[0])

    def test_roundtrip_with_gray_config(self, tmp_path):
        path = tmp_path / "out.mp4"
        with self._make(path) as w:
            for f in _gray(5):
                w.write(f)
        assert len(_read_frames(path, "gray")) == 5

    def test_roundtrip_with_color_config(self, tmp_path):
        path = tmp_path / "out.mp4"
        cfg = EncoderConfig(codec="libx264", pix_fmt="yuv420p",
                            options={"crf": "28", "preset": "ultrafast"})
        with PyAVVideoFileWriter(path, size=(W, H), fps=FPS,
                                 encoder_config=cfg, input_pix_fmt="bgr24") as w:
            for f in _color(3):
                w.write(f)
        assert len(_read_frames(path, "bgr24")) == 3

    def test_invalid_size_raises(self, tmp_path):
        cfg = EncoderConfig(codec="libx264", pix_fmt="gray",
                            options={"crf": "28", "preset": "ultrafast"})
        with pytest.raises(ValueError):
            PyAVVideoFileWriter(tmp_path / "out.mp4", size=(0, H), fps=FPS,
                                encoder_config=cfg)

    def test_invalid_fps_raises(self, tmp_path):
        cfg = EncoderConfig(codec="libx264", pix_fmt="gray",
                            options={"crf": "28", "preset": "ultrafast"})
        with pytest.raises(ValueError):
            PyAVVideoFileWriter(tmp_path / "out.mp4", size=(W, H), fps=0,
                                encoder_config=cfg)






