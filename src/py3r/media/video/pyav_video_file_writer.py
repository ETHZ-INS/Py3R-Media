from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import av
import numpy as np

from py3r.media.types import HasImage

QualityName = Literal["very_low", "low", "medium", "high", "very_high", "lossless"]


class PyAVVideoFileWriter:
    """
    Write video frames to a file using PyAV (libav*) — no ffmpeg executable needed.

    Mirrors the ``FFmpegVideoFileWriter`` interface exactly so the two are
    drop-in replacements for each other.

    Parameters
    ----------
    path : str | Path
        Output file path.  The container format is inferred from the extension
        (e.g. ``.mp4``, ``.mkv``, ``.avi``).
    size : (width, height)
        Frame dimensions in pixels.
    fps : float
        Nominal frame-rate.
    grayscale : bool
        If ``True``, ``write()`` expects ``(H, W)`` / ``(H, W, 1)`` uint8 arrays.
        If ``False``, ``write()`` expects ``(H, W, 3)`` uint8 BGR arrays.
    quality : QualityName
        One of ``"very_low"``, ``"low"``, ``"medium"``, ``"high"``,
        ``"very_high"``, or ``"lossless"``.

        +------------+-----+-----------+
        | quality    | crf | preset    |
        +============+=====+===========+
        | very_low   |  36 | ultrafast |
        | low        |  32 | ultrafast |
        | medium     |  28 | veryfast  |
        | high       |  22 | medium    |
        | very_high  |  18 | fast      |
        | lossless   |   0 | veryslow  |
        +------------+-----+-----------+

        ``"lossless"`` uses ``libx264rgb`` (color) or ``libx264`` with
        ``pix_fmt=gray`` (grayscale) so that no information is thrown away.
        Prefer ``.mkv`` for lossless because MP4 chokes on ``bgr24``/``gray``
        pixel formats in some players.
    extra_options : dict[str, str] | None
        Additional codec private options merged on top of the quality preset
        (e.g. ``{"tune": "film"}``).

    Notes
    -----
    * Frames are encoded in-order with monotonically increasing PTS.
    * Gray input is reformatted to ``yuv420p`` for all non-lossless qualities
      to maximise player compatibility.
    * ``close()`` flushes the encoder and finalises container headers — always
      call it (or use the writer as a context manager).
    """

    QUALITY_PRESETS: Dict[str, Dict[str, str]] = {
        "very_low":  {"crf": "36", "preset": "ultrafast"},
        "low":       {"crf": "32", "preset": "ultrafast"},
        "medium":    {"crf": "28", "preset": "veryfast"},
        "high":      {"crf": "22", "preset": "medium"},
        "very_high": {"crf": "18", "preset": "fast"},
    }

    def __init__(
        self,
        path: Union[Path, str],
        size: Tuple[int, int],
        fps: float,
        *,
        grayscale: bool = False,
        quality: QualityName = "medium",
        extra_options: Optional[Dict[str, str]] = None,
    ) -> None:
        self._path = Path(path)
        self._w, self._h = int(size[0]), int(size[1])
        self._fps = float(fps)
        self._grayscale = bool(grayscale)
        self._quality = quality
        self._extra_options: Dict[str, str] = extra_options or {}

        self._container: Optional[av.container.OutputContainer] = None
        self._stream = None
        self._frame_count: int = 0
        self._closed: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if not self._closed:
            return

        self._container = av.open(str(self._path), mode="w")

        # ---- codec + pixel format selection ----
        if self._quality == "lossless":
            if self._grayscale:
                # libx264 natively encodes gray without chroma waste
                codec_name = "libx264"
                enc_pix_fmt = "gray"
            else:
                # libx264rgb preserves BGR bitwise
                codec_name = "libx264rgb"
                enc_pix_fmt = "bgr24"
            options: Dict[str, str] = {"crf": "0", "preset": "veryslow"}
        else:
            codec_name = "libx264"
            # yuv420p is the most universally accepted h264 pixel format
            enc_pix_fmt = "yuv420p"
            options = dict(self.QUALITY_PRESETS[self._quality])

        options.update(self._extra_options)

        # Use a rational fps to avoid floating-point drift
        rate = Fraction(self._fps).limit_denominator(1001)

        stream = self._container.add_stream(codec_name, rate=rate)
        stream.width = self._w
        stream.height = self._h
        stream.pix_fmt = enc_pix_fmt
        stream.options = options

        self._stream = stream
        # For lossless grey, signal full colour range (AVCOL_RANGE_JPEG) so that
        # decoders don't apply the limited-range rescale (Y∈[16,235] → [0,255])
        # that would corrupt the lossless guarantee.
        if self._quality == "lossless" and self._grayscale:
            try:
                self._stream.codec_context.color_range = 2  # AVCOL_RANGE_JPEG
            except Exception:
                pass
        self._frame_count = 0
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._stream is not None:
                # Flush any buffered frames from the encoder
                for packet in self._stream.encode(None):
                    self._container.mux(packet)  # type: ignore[union-attr]
            if self._container is not None:
                self._container.close()
        finally:
            self._stream = None
            self._container = None
            self._closed = True

    def __enter__(self) -> "PyAVVideoFileWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, frame: Union[HasImage, np.ndarray]) -> None:
        """
        Write a single frame.

        Parameters
        ----------
        frame : numpy.ndarray or HasImage
            * Color (``grayscale=False``): shape ``(H, W, 3)``, dtype ``uint8``,
              **BGR** channel order (OpenCV-native).
            * Grayscale (``grayscale=True``): shape ``(H, W)`` or ``(H, W, 1)``,
              dtype ``uint8``.

        Raises
        ------
        RuntimeError
            Writer has not been opened.
        ValueError
            Frame dtype, shape, or channel count does not match the writer
            configuration.
        """
        if self._closed or self._stream is None:
            raise RuntimeError("PyAVVideoFileWriter is not open — call .open() first.")

        img = frame if isinstance(frame, np.ndarray) else frame.img

        if img.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8, got {img.dtype}")

        # ---- shape normalisation + validation ----
        if self._grayscale:
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
            if img.ndim != 2:
                raise ValueError(
                    f"Grayscale mode expects shape (H, W) or (H, W, 1), got {img.shape}"
                )
            src_fmt = "gray"
        else:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Color mode expects shape (H, W, 3), got {img.shape}"
                )
            src_fmt = "bgr24"

        if img.shape[0] != self._h or img.shape[1] != self._w:
            raise ValueError(
                f"Frame size mismatch: expected ({self._h}, {self._w}), "
                f"got {img.shape[:2]}"
            )

        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        # ---- build PyAV frame ----
        av_frame = av.VideoFrame.from_ndarray(img, format=src_fmt)
        av_frame.pts = self._frame_count
        self._frame_count += 1

        # Reformat to encoder pixel format (e.g. gray → yuv420p)
        enc_fmt: str = self._stream.codec_context.pix_fmt
        if av_frame.format.name != enc_fmt:
            av_frame = av_frame.reformat(format=enc_fmt)

        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """``True`` if the writer has been opened and not yet closed."""
        return not self._closed

