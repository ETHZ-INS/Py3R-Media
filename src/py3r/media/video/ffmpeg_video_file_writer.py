from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np

from py3r.media.types import HasImage

QualityName = Literal["very_low", "low", "medium", "high", "very_high", "lossless"]


class FFmpegVideoFileWriter:
    """
    Stream raw frames (gray or BGR) to ffmpeg/libx264.

    Parameters
    ----------
    path : str | Path
        Output video file path (e.g., 'out.mp4' or 'out.mkv').
    size : (width, height)
        Frame size in pixels.
    fps : float
        Nominal framerate; used for input timestamps.
    grayscale : bool
        If True, expect BGR frames (HxWx3). If False, expect grayscale (HxW or HxWx1).
    quality : QualityName
        One of: "very_low","low","medium","high","very_high","lossless".
        See QUALITY_PRESETS for mapping. "lossless" uses libx264rgb.
    ffmpeg_path : str
        Optional custom ffmpeg binary path.
    extra_output_args : list[str] | None
        Extra flags appended after codec settings (e.g., ["-movflags","+faststart"]).

    Notes
    -----
    - For "lossless", prefer `.mkv` container (MP4 may not like RGB/BGR).
    - Writes are zero-copy from NumPy where possible via memoryview.
    """

    QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
        "very_low":  dict(crf=36, preset="ultrafast"),
        "low":       dict(crf=32, preset="ultrafast"),
        "medium":    dict(crf=28, preset="veryfast"),
        "high":      dict(crf=22, preset="medium"),
        "very_high": dict(crf=18, preset="fast"),
        # "lossless" handled specially (libx264rgb + crf=0 + pix_fmt=bgr24)
    }

    def __init__(
        self,
        path: Path | str,
        size: Tuple[int, int],
        fps: float,
        *,
        grayscale: bool = False,
        quality: QualityName = "medium",
        ffmpeg_path: str = "ffmpeg",
        extra_output_args: Optional[list[str]] = None,
    ):
        self._path = str(path)
        self._w, self._h = int(size[0]), int(size[1])
        self._fps = float(fps)
        self._grayscale = bool(grayscale)
        self._quality = quality
        self._ffmpeg = ffmpeg_path
        self._extra = extra_output_args or []

        self._proc: Optional[subprocess.Popen] = None
        self._frame_nbytes = self._w * self._h * (1 if self._grayscale else 3)

        # Preallocate a reusable buffer for quick shape checks if needed
        self._closed = True

    # ----------------- lifecycle -----------------

    def open(self) -> None:
        if not self._closed:
            return

        # Input spec (rawvideo on stdin)
        in_pix_fmt = "gray" if self._grayscale else "bgr24"

        # Output codec + pixel format
        if self._quality == "lossless":
            # libx264rgb wants RGB/BGR input and keeps it; use bgr24 for OpenCV-native
            vcodec = "libx264rgb"
            crf = "0"
            preset = "veryslow"  # can tweak; lossless often pairs with slower preset
            out_pix_fmt = "bgr24"
            enc_args = ["-c:v", vcodec, "-crf", crf, "-preset", preset, "-pix_fmt", out_pix_fmt]
        else:
            q = self.QUALITY_PRESETS[self._quality]
            vcodec = "libx264"
            crf = str(q["crf"])
            preset = q["preset"]
            # yuv420p is most compatible for playback
            out_pix_fmt = "yuv420p"
            enc_args = ["-c:v", vcodec, "-crf", crf, "-preset", preset, "-pix_fmt", out_pix_fmt]

        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",  # overwrite output
            # Input (rawvideo from stdin)
            "-f", "rawvideo",
            "-pix_fmt", in_pix_fmt,
            "-s:v", f"{self._w}x{self._h}",
            "-r", f"{self._fps:g}",
            "-i", "pipe:0",
            # Encoder / output
            *enc_args,
            *self._extra,
            self._path,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # keep for debugging if needed
            bufsize=0,
        )
        if self._proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin pipe.")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            if self._proc:
                self._proc.wait(timeout=5.0)
                # Optional: inspect stderr for errors
                if self._proc.returncode not in (0, None):
                    err = self._proc.stderr.read().decode("utf-8", errors="ignore") if self._proc.stderr else ""
                    raise RuntimeError(f"ffmpeg exited with code {self._proc.returncode}\n{err}")
        finally:
            self._proc = None
            self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ----------------- writing -----------------

    def write(self, frame: HasImage | np.ndarray) -> None:
        """
        Write a single frame.

        frame:
            - color=True:   shape (H, W, 3), dtype=uint8, **BGR** order.
            - color=False:  shape (H, W) or (H, W, 1), dtype=uint8.
        """

        frame = frame if isinstance(frame, np.ndarray) else frame.img

        if self._closed:
            raise RuntimeError("FFmpegVideoWriter is not open. Call .open() first.")

        if frame.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype=uint8, got {frame.dtype}")

        # Validate shape
        if self._grayscale:
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame.reshape((frame.shape[0], frame.shape[1]))
            if frame.ndim != 2:
                raise ValueError(f"Grayscale requires shape (H, W) or (H, W, 1), got {frame.shape}")
            if frame.shape[0] != self._h or frame.shape[1] != self._w:
                raise ValueError(f"Frame size mismatch. Expected {(self._h, self._w)}, got {frame.shape[:2]}")
        else:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Color=True requires shape (H, W, 3), got {frame.shape}")
            if frame.shape[0] != self._h or frame.shape[1] != self._w:
                raise ValueError(f"Frame size mismatch. Expected {(self._h, self._w)}, got {frame.shape[:2]}")

        # Ensure C-contiguous
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        mv = memoryview(frame)

        # Write raw bytes
        try:
            self._proc.stdin.write(mv)  # type: ignore[arg-type]
        except BrokenPipeError:
            err = self._proc.stderr.read().decode("utf-8", errors="ignore") if self._proc and self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg pipe closed. {err}") from None

    # ----------------- helpers -----------------

    @property
    def is_open(self) -> bool:
        return not self._closed
