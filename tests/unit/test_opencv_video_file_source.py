"""
Unit tests for OpenCVVideoFileSource.

All tests use synthetic MP4 files produced by the session fixtures in
``tests/conftest.py`` — no real footage is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from py3r.media.video.opencv_video_file_source import (
    EndOfStreamError,
    OpenCVVideoFileSource,
)
from tests.conftest import VIDEO_FPS, VIDEO_H, VIDEO_N, VIDEO_W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_all(src: OpenCVVideoFileSource) -> list:
    """Drain *src* until EndOfStreamError; return all VideoFrames."""
    frames = []
    try:
        while True:
            frames.append(src.read())
    except EndOfStreamError:
        pass
    return frames


# ---------------------------------------------------------------------------
# Probe / metadata  (no open() required)
# ---------------------------------------------------------------------------

class TestProbe:
    def test_size(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        assert src.get_size() == (VIDEO_W, VIDEO_H)

    def test_fps(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        assert src.get_fps() == pytest.approx(VIDEO_FPS, abs=1.0)

    def test_num_frames(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        assert src.get_num_frames() == VIDEO_N

    def test_capability_flags(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        assert src.has_size()
        assert src.has_fps()
        assert src.has_num_frames()
        assert src.has_timing()
        assert src.is_seekable()
        assert not src.is_open()

    def test_loop_hides_num_frames(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True, loop=True)
        assert not src.has_num_frames()
        assert src.get_num_frames() is None

    def test_num_channels_grayscale(self, synthetic_gray_mp4):
        assert OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True).get_num_channels() == 1

    def test_num_channels_color(self, synthetic_color_mp4):
        assert OpenCVVideoFileSource(synthetic_color_mp4, grayscale=False).get_num_channels() == 3

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot open"):
            OpenCVVideoFileSource(tmp_path / "no_such_file.mp4")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_open_after_open(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        assert src.is_open()
        src.close()

    def test_is_closed_after_close(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.close()
        assert not src.is_open()

    def test_double_close_is_safe(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.close()
        src.close()  # must not raise

    def test_reopen_resets_index(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.read()
        src.read()
        src.close()
        src.open()
        frame = src.read()
        src.close()
        assert frame.frame_index == 0

    def test_read_before_open_raises(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        with pytest.raises(RuntimeError, match="not open"):
            src.read()

    def test_seek_before_open_raises(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        with pytest.raises(RuntimeError, match="not open"):
            src.seek(5)


# ---------------------------------------------------------------------------
# Frame content and shape
# ---------------------------------------------------------------------------

class TestFrameContent:
    def test_grayscale_shape_and_dtype(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frame = src.read()
        src.close()
        assert frame.img.shape == (VIDEO_H, VIDEO_W)
        assert frame.img.dtype == np.uint8

    def test_color_shape_and_dtype(self, synthetic_color_mp4):
        src = OpenCVVideoFileSource(synthetic_color_mp4, grayscale=False)
        src.open()
        frame = src.read()
        src.close()
        assert frame.img.shape == (VIDEO_H, VIDEO_W, 3)
        assert frame.img.dtype == np.uint8

    def test_grayscale_forced_from_color_file(self, synthetic_color_mp4):
        """grayscale=True on a color file must return (H, W) frames."""
        src = OpenCVVideoFileSource(synthetic_color_mp4, grayscale=True)
        src.open()
        frame = src.read()
        src.close()
        assert frame.img.ndim == 2

    def test_brightness_increases_across_frames(self, synthetic_gray_mp4):
        """The synthetic file has ascending brightness; verify ordering survives decode."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        means = [f.img.mean() for f in frames]
        # First quarter must be darker than last quarter
        assert np.mean(means[: VIDEO_N // 4]) < np.mean(means[-VIDEO_N // 4 :])

    def test_first_frame_is_darkest(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        means = [f.img.mean() for f in frames]
        assert means[0] == min(means)

    def test_last_frame_is_brightest(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        means = [f.img.mean() for f in frames]
        assert means[-1] == max(means)


# ---------------------------------------------------------------------------
# Frame index
# ---------------------------------------------------------------------------

class TestFrameIndex:
    def test_indices_are_consecutive_from_zero(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        assert [f.frame_index for f in frames] == list(range(VIDEO_N))

    def test_total_frame_count(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        assert len(frames) == VIDEO_N


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamps_are_non_negative(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        assert all(f.timestamp >= 0.0 for f in frames)

    def test_timestamps_are_non_decreasing(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        ts = [f.timestamp for f in frames]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))

    def test_final_timestamp_reasonable(self, synthetic_gray_mp4):
        """Last frame timestamp should be roughly (N-1) / fps seconds."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        frames = _read_all(src)
        src.close()
        expected = (VIDEO_N - 1) / VIDEO_FPS
        assert frames[-1].timestamp == pytest.approx(expected, abs=0.5)


# ---------------------------------------------------------------------------
# End-of-stream behaviour
# ---------------------------------------------------------------------------

class TestEndOfStream:
    def test_raises_eos_not_none(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        for _ in range(VIDEO_N):
            src.read()
        with pytest.raises(EndOfStreamError):
            src.read()
        src.close()

    def test_eos_is_exception_not_none(self, synthetic_gray_mp4):
        """The old interface returned None; the new one must never do that."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        result = None
        try:
            for _ in range(VIDEO_N + 1):
                result = src.read()
        except EndOfStreamError:
            pass
        finally:
            src.close()
        # result should be a VideoFrame from the last successful read, not None
        assert result is not None
        assert result.frame_index == VIDEO_N - 1


# ---------------------------------------------------------------------------
# Loop behaviour
# ---------------------------------------------------------------------------

class TestLoop:
    def test_loop_continues_past_eos(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True, loop=True)
        src.open()
        # Read 1.5x the file length
        extra = VIDEO_N + VIDEO_N // 2
        frames = [src.read() for _ in range(extra)]
        src.close()
        assert len(frames) == extra

    def test_loop_frame_indices_keep_incrementing(self, synthetic_gray_mp4):
        """After wrapping, _idx must continue from where it left off — not reset to 0."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True, loop=True)
        src.open()
        extra = VIDEO_N + 5
        frames = [src.read() for _ in range(extra)]
        src.close()
        assert [f.frame_index for f in frames] == list(range(extra))

    def test_loop_repeats_content(self, synthetic_gray_mp4):
        """Frames VIDEO_N .. 2*VIDEO_N-1 should have the same pixel content as 0 .. VIDEO_N-1."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True, loop=True)
        src.open()
        frames = [src.read() for _ in range(VIDEO_N * 2)]
        src.close()
        for i in range(VIDEO_N):
            # Brightness ordering must repeat (exact values may differ due to OCV backend)
            assert frames[i].img.mean() == pytest.approx(
                frames[i + VIDEO_N].img.mean(), abs=2.0
            )


# ---------------------------------------------------------------------------
# Seek
# ---------------------------------------------------------------------------

class TestSeek:
    def test_seek_sets_frame_index(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.seek(10)
        frame = src.read()
        src.close()
        assert frame.frame_index == 10

    def test_seek_delivers_correct_content(self, synthetic_gray_mp4):
        """Seeking to frame 20 should deliver a brighter frame than frame 5."""
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.seek(5)
        bright5 = src.read().img.mean()
        src.seek(20)
        bright20 = src.read().img.mean()
        src.close()
        assert bright20 > bright5

    def test_seek_to_zero_restarts(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        # Advance to the middle
        for _ in range(15):
            src.read()
        src.seek(0)
        frame = src.read()
        src.close()
        assert frame.frame_index == 0

    def test_consecutive_reads_after_seek(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.seek(15)
        f1 = src.read()
        f2 = src.read()
        src.close()
        assert f1.frame_index == 15
        assert f2.frame_index == 16

    def test_eos_still_raised_after_seek_near_end(self, synthetic_gray_mp4):
        src = OpenCVVideoFileSource(synthetic_gray_mp4, grayscale=True)
        src.open()
        src.seek(VIDEO_N - 1)
        src.read()  # last frame
        with pytest.raises(EndOfStreamError):
            src.read()
        src.close()

