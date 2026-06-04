from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import av
import numpy as np

from py3r.media.types import HasImage

QualityName = Literal["very_low", "low", "medium", "high", "very_high", "lossless"]


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """
    Raw encoder configuration for :class:`PyAVStreamWriter`.

    Parameters
    ----------
    codec : str
        libav codec name (e.g. ``"libx264"``, ``"libx264rgb"``,
        ``"libx265"``, ``"libvpx-vp9"``).
    pix_fmt : str
        Output pixel format (e.g. ``"yuv420p"``, ``"gray"``, ``"bgr24"``).
    options : dict[str, str]
        Codec private options passed verbatim to the encoder
        (e.g. ``{"crf": "28", "preset": "veryfast"}``).
    full_range : bool
        When ``True``, signals ``AVCOL_RANGE_JPEG`` (full / PC range) on the
        stream's codec context.  Required for lossless grayscale to prevent
        decoders applying limited-range rescaling (Y∈[16,235] → [0,255]).
    """

    codec: str
    pix_fmt: str
    options: Dict[str, str] = field(default_factory=dict)
    full_range: bool = False


_QUALITY_TABLE: Dict[str, Dict[str, str]] = {
    "very_low":  {"crf": "36", "preset": "ultrafast"},
    "low":       {"crf": "32", "preset": "ultrafast"},
    "medium":    {"crf": "28", "preset": "veryfast"},
    "high":      {"crf": "22", "preset": "medium"},
    "very_high": {"crf": "18", "preset": "fast"},
    "lossless":  {"crf": "0",  "preset": "veryslow"},
}


def resolve_encoder_config(
    quality: QualityName,
    grayscale: bool = False,
    extra_options: Optional[Dict[str, str]] = None,
) -> EncoderConfig:
    """
    Map a *quality* name and color mode to a concrete :class:`EncoderConfig`.

    +------------+-----+-----------+------------------+
    | quality    | crf | preset    | codec            |
    +============+=====+===========+==================+
    | very_low   |  36 | ultrafast | libx264          |
    | low        |  32 | ultrafast | libx264          |
    | medium     |  28 | veryfast  | libx264          |
    | high       |  22 | medium    | libx264          |
    | very_high  |  18 | fast      | libx264          |
    | lossless   |   0 | veryslow  | libx264 / x264rgb|
    +------------+-----+-----------+------------------+

    Parameters
    ----------
    quality : QualityName
        One of the keys in the table above.
    grayscale : bool
        Selects ``gray`` pixel format and ``libx264`` with full-range signalling
        for lossless, or ``yuv420p`` / ``bgr24`` otherwise.
    extra_options : dict[str, str] | None
        Additional codec options merged on top of the quality preset.

    Returns
    -------
    EncoderConfig
    """
    if quality not in _QUALITY_TABLE:
        raise ValueError(
            f"Unknown quality {quality!r}. "
            f"Valid values: {list(_QUALITY_TABLE)}"
        )

    if quality == "lossless":
        if grayscale:
            codec, pix_fmt, full_range = "libx264", "gray", True
        else:
            codec, pix_fmt, full_range = "libx264rgb", "bgr24", False
    else:
        codec, pix_fmt, full_range = "libx264", "yuv420p", False

    options = dict(_QUALITY_TABLE[quality])
    if extra_options:
        options.update(extra_options)

    return EncoderConfig(codec=codec, pix_fmt=pix_fmt, options=options, full_range=full_range)


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

class PyAVStreamWriter:
    """
    Write video frames to a file using a caller-supplied :class:`EncoderConfig`.

    This is the low-level class.  It does not know about quality names or
    presets — those are resolved before construction.  Any codec and pixel
    format supported by the installed libav build can be used.

    Parameters
    ----------
    path : str | Path
        Output file path.  The container format is inferred from the extension.
    size : (width, height)
        Frame dimensions in pixels.
    fps : float
        Nominal frame-rate.
    encoder_config : EncoderConfig
        Codec name, pixel format, and encoder options.
    grayscale : bool
        If ``True``, :meth:`write` expects ``(H, W)`` / ``(H, W, 1)`` uint8
        arrays.  If ``False``, it expects ``(H, W, 3)`` uint8 BGR arrays.

    Notes
    -----
    * ``close()`` flushes the encoder and finalises container headers — always
      call it (or use the writer as a context manager).
    """

    def __init__(
        self,
        path: Union[Path, str],
        size: Tuple[int, int],
        fps: float,
        encoder_config: EncoderConfig,
        *,
        grayscale: bool = False,
    ) -> None:
        self._path = Path(path)
        self._w, self._h = int(size[0]), int(size[1])
        self._fps = float(fps)
        self._encoder_config = encoder_config
        self._grayscale = bool(grayscale)

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

        cfg = self._encoder_config
        self._container = av.open(str(self._path), mode="w")

        rate = Fraction(self._fps).limit_denominator(1001)
        stream = self._container.add_stream(cfg.codec, rate=rate)
        stream.width = self._w
        stream.height = self._h
        stream.pix_fmt = cfg.pix_fmt
        stream.options = dict(cfg.options)

        if cfg.full_range:
            try:
                stream.codec_context.color_range = 2  # AVCOL_RANGE_JPEG
            except Exception:
                pass

        self._stream = stream
        self._frame_count = 0
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._stream is not None:
                for packet in self._stream.encode(None):
                    self._container.mux(packet)  # type: ignore[union-attr]
            if self._container is not None:
                self._container.close()
        finally:
            self._stream = None
            self._container = None
            self._closed = True

    def __enter__(self) -> "PyAVStreamWriter":
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
            raise RuntimeError("PyAVStreamWriter is not open — call .open() first.")

        img = frame if isinstance(frame, np.ndarray) else frame.img

        if img.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8, got {img.dtype}")

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

        av_frame = av.VideoFrame.from_ndarray(img, format=src_fmt)
        av_frame.pts = self._frame_count
        self._frame_count += 1

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


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class PyAVVideoFileWriter:
    """
    Convenience wrapper around :class:`PyAVStreamWriter` that resolves a
    *quality* name to an :class:`EncoderConfig` via :func:`resolve_encoder_config`.

    For full control over the codec and options (e.g. a different codec or
    custom options not representable as a quality name), construct a
    :class:`PyAVStreamWriter` directly with a hand-built :class:`EncoderConfig`.

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
        If ``True``, :meth:`write` expects ``(H, W)`` / ``(H, W, 1)`` uint8
        arrays.  If ``False``, it expects ``(H, W, 3)`` uint8 BGR arrays.
    quality : QualityName
        Resolved by :func:`resolve_encoder_config`.
    extra_options : dict[str, str] | None
        Additional codec options merged on top of the quality preset.
    """

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
        cfg = resolve_encoder_config(quality, grayscale, extra_options)
        self._writer = PyAVStreamWriter(path, size, fps, cfg, grayscale=grayscale)

    def open(self) -> None:
        self._writer.open()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "PyAVVideoFileWriter":
        self._writer.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._writer.close()

    def write(self, frame: Union[HasImage, np.ndarray]) -> None:
        self._writer.write(frame)

    @property
    def is_open(self) -> bool:
        return self._writer.is_open

