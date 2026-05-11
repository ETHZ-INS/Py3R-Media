"""
Top-level pytest fixtures shared by all test modules.

Synthetic video files
---------------------
Rather than bundling real footage, the fixtures here generate small,
deterministic MP4 files at session start using PyAV directly (no subprocess,
no dependency on the classes under test).

Frame content
~~~~~~~~~~~~~
Each frame is a solid fill with a single gray/colour value that increases
linearly from 0 → 255 across the N frames.  Solid uniform frames round-trip
through YUV conversion exactly (no chroma, so Y == the original gray value),
which lets tests assert pixel content precisely even with lossy containers.

Layout
~~~~~~
VIDEO_W × VIDEO_H pixels, VIDEO_N frames at VIDEO_FPS fps.
Stored as H.264/yuv420p inside an MP4 so cv2.VideoCapture can parse them
reliably on all platforms.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CLI options  (must live here so they are registered before any collection)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-webcam",
        action="store_true",
        default=False,
        help="Fail (instead of skip) when no USB webcam is detected.",
    )
    parser.addoption(
        "--require-pylon",
        action="store_true",
        default=False,
        help="Fail (instead of skip) when no Basler/Pylon camera is detected.",
    )
    parser.addoption(
        "--webcam-name",
        default=None,
        metavar="NAME",
        help=(
            "Platform device name passed to PyAVWebcamSource.  "
            "Windows DirectShow example: 'Integrated Camera'.  "
            "Linux V4L2 example: '/dev/video0'.  "
            "Required to run PyAVWebcamSource hardware tests."
        ),
    )

# ---------------------------------------------------------------------------
# Constants shared across unit tests
# ---------------------------------------------------------------------------

VIDEO_W = 64
VIDEO_H = 48
VIDEO_N = 30       # frames per synthetic file
VIDEO_FPS = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frames(n: int, width: int, height: int, grayscale: bool) -> list[np.ndarray]:
    """Return *n* uint8 frames with linearly increasing brightness.

    Grayscale:  shape (H, W)        — value = round(255 * i / (n-1))
    Color BGR:  shape (H, W, 3)     — blue channel encodes brightness,
                                       green & red are 0.
    Frames are solid fills, ensuring exact round-trips through yuv420p.
    """
    frames = []
    for i in range(n):
        val = round(255 * i / max(n - 1, 1))
        if grayscale:
            frames.append(np.full((height, width), val, dtype=np.uint8))
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = val  # blue
            frames.append(img)
    return frames


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float, src_fmt: str) -> None:
    """Write *frames* to *path* as H.264/yuv420p MP4 via PyAV."""
    rate = Fraction(fps).limit_denominator(1001)
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        # crf=0 is bitwise lossless *within YUV space*; solid-fill frames
        # survive the gray→yuv→gray round-trip exactly.
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i, img in enumerate(frames):
            av_frame = av.VideoFrame.from_ndarray(img, format=src_fmt)
            av_frame = av_frame.reformat(format="yuv420p")
            av_frame.pts = i
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)


# ---------------------------------------------------------------------------
# Session-scoped fixtures  (created once, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_gray_mp4(tmp_path_factory) -> Path:
    """30-frame 64×48 grayscale MP4 at 30 fps."""
    path = tmp_path_factory.mktemp("video") / "gray_30f.mp4"
    frames = make_frames(VIDEO_N, VIDEO_W, VIDEO_H, grayscale=True)
    _write_mp4(path, frames, VIDEO_FPS, src_fmt="gray")
    return path


@pytest.fixture(scope="session")
def synthetic_color_mp4(tmp_path_factory) -> Path:
    """30-frame 64×48 BGR color MP4 at 30 fps."""
    path = tmp_path_factory.mktemp("video") / "color_30f.mp4"
    frames = make_frames(VIDEO_N, VIDEO_W, VIDEO_H, grayscale=False)
    _write_mp4(path, frames, VIDEO_FPS, src_fmt="bgr24")
    return path
