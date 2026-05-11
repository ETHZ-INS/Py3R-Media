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
from py3r.media.video.pyav_video_file_writer import PyAVVideoFileWriter

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
                 quality: str = "medium", **kw) -> PyAVVideoFileWriter:
    return PyAVVideoFileWriter(path, size=(W, H), fps=FPS,
                               grayscale=grayscale, quality=quality, **kw)


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
        with _make_writer(path, quality="lossless") as w:
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
        with _make_writer(path, quality="lossless",
                          extra_options={"bf": "0"}) as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "gray")
        assert len(decoded) == len(frames)
        for orig, dec in zip(frames, decoded):
            np.testing.assert_array_equal(orig, dec)

    def test_lossless_color_exact(self, tmp_path):
        path = tmp_path / "out.mkv"
        frames = _color(5)
        with _make_writer(path, grayscale=False, quality="lossless") as w:
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
        with _make_writer(path, quality="medium") as w:
            for f in frames:
                w.write(f)
        decoded = _read_frames(path, "gray")
        assert len(decoded) == len(frames)
        for orig, dec in zip(frames, decoded):
            np.testing.assert_allclose(dec.astype(int), orig.astype(int), atol=1)

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
# TestQuality
# ---------------------------------------------------------------------------

class TestQuality:
    @pytest.mark.parametrize("quality", [
        "very_low", "low", "medium", "high", "very_high",
    ])
    def test_lossy_quality_produces_valid_file(self, tmp_path, quality):
        path = tmp_path / f"out_{quality}.mp4"
        with _make_writer(path, quality=quality) as w:
            w.write(_gray(1)[0])
        decoded = _read_frames(path, "gray")
        assert len(decoded) == 1

    def test_lossless_quality_produces_valid_gray_file(self, tmp_path):
        path = tmp_path / "out_lossless.mkv"
        with _make_writer(path, quality="lossless") as w:
            w.write(_gray(1)[0])
        decoded = _read_frames(path, "gray")
        assert len(decoded) == 1

    def test_lossless_quality_produces_valid_color_file(self, tmp_path):
        path = tmp_path / "out_lossless.mkv"
        with _make_writer(path, grayscale=False, quality="lossless") as w:
            w.write(_color(1)[0])
        decoded = _read_frames(path, "bgr24")
        assert len(decoded) == 1

    def test_extra_options_are_merged(self, tmp_path):
        """extra_options must not break encoding — just verify file is valid."""
        path = tmp_path / "out.mp4"
        with _make_writer(path, extra_options={"tune": "grain"}) as w:
            w.write(_gray(1)[0])
        decoded = _read_frames(path, "gray")
        assert len(decoded) == 1





