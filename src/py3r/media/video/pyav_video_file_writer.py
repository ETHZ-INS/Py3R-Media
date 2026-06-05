from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import av
import numpy as np

from py3r.media.types import HasImage


InputPixFmt = Literal[
    "gray",
    "bgr24",
    "rgb24",
    "bgra",
    "rgba",
]


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderConfig:
    """
    Encoder / stream-side configuration for PyAVStreamWriter.

    This describes how frames are encoded, not the format of the NumPy arrays
    passed to write().

    Parameters
    ----------
    codec:
        FFmpeg / libav codec name, for example "libx264", "libx264rgb",
        "libx265", "libvpx-vp9".
    pix_fmt:
        Pixel format configured on the output stream / encoder, for example
        "yuv420p", "gray", "bgr24".
    options:
        Codec private options passed to the encoder, for example
        {"crf": "22", "preset": "medium"}.
    codec_context_attrs:
        Extra attributes set on stream.codec_context before encoding starts.
        Useful for things like color_range, thread_count, color_primaries, etc.
    """

    codec: str
    pix_fmt: str
    options: Mapping[str, str] = field(default_factory=dict)
    codec_context_attrs: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frame validation / normalization
# ---------------------------------------------------------------------------

_CHANNELS_BY_INPUT_PIX_FMT: Dict[str, int] = {
    "bgr24": 3,
    "rgb24": 3,
    "bgra": 4,
    "rgba": 4,
}


def _as_ndarray(frame: Union[HasImage, np.ndarray]) -> np.ndarray:
    return frame if isinstance(frame, np.ndarray) else frame.img


def _validate_and_normalize_frame(
    img: np.ndarray,
    *,
    input_pix_fmt: str,
    expected_size: Tuple[int, int],
) -> np.ndarray:
    """
    Validate the frame according to the writer's input contract.

    expected_size is (width, height), matching the writer constructor.
    """
    expected_w, expected_h = expected_size

    if img.dtype != np.uint8:
        raise ValueError(f"Expected dtype=uint8, got {img.dtype}")

    if img.shape[0] != expected_h or img.shape[1] != expected_w:
        raise ValueError(
            f"Frame size mismatch: expected ({expected_h}, {expected_w}), "
            f"got {img.shape[:2]}"
        )

    if input_pix_fmt == "gray":
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])

        if img.ndim != 2:
            raise ValueError(
                "input_pix_fmt='gray' expects shape (H, W) or (H, W, 1), "
                f"got {img.shape}"
            )

    elif input_pix_fmt in _CHANNELS_BY_INPUT_PIX_FMT:
        expected_channels = _CHANNELS_BY_INPUT_PIX_FMT[input_pix_fmt]

        if img.ndim != 3 or img.shape[2] != expected_channels:
            raise ValueError(
                f"input_pix_fmt={input_pix_fmt!r} expects shape "
                f"(H, W, {expected_channels}), got {img.shape}"
            )

    else:
        # Unknown / advanced PyAV pixel format.
        # We still check dtype and size, but leave shape validation to PyAV.
        if img.ndim < 2:
            raise ValueError(
                f"Expected at least 2D image data for {input_pix_fmt!r}, "
                f"got {img.shape}"
            )

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    return img


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

class PyAVVideoFileWriter:
    """
    Minimal video writer around a PyAV output container and video stream.

    The writer owns three separate concepts:

    1. Input frame format:
       input_pix_fmt describes the NumPy arrays passed to write().

    2. Encoder configuration:
       encoder_config describes the PyAV stream / codec.

    3. Lifecycle:
       open(), write(), close().

    Parameters
    ----------
    path:
        Output file path. Container format is normally inferred from extension.
    size:
        Frame size as (width, height).
    fps:
        Nominal frame rate.
    encoder_config:
        EncoderConfig describing codec, output pixel format, codec options,
        and codec context attributes.
    input_pix_fmt:
        Pixel format of frames passed to write(), for example "bgr24",
        "rgb24", or "gray".
    container_format:
        Optional explicit container format passed to av.open(..., format=...).
        Usually unnecessary if path has a normal extension.
    open_options:
        Extra keyword arguments passed to av.open(..., mode="w", **open_options).
    configure_stream:
        Optional callback called after the stream has been created and basic
        attributes have been set, but before any frame is encoded.
    """

    def __init__(
        self,
        path: Union[Path, str],
        size: Tuple[int, int],
        fps: float,
        encoder_config: EncoderConfig,
        *,
        input_pix_fmt: str = "bgr24",
        container_format: Optional[str] = None,
        open_options: Optional[Mapping[str, Any]] = None,
        configure_stream: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._path = Path(path)
        self._w = int(size[0])
        self._h = int(size[1])
        self._fps = float(fps)

        if self._w <= 0 or self._h <= 0:
            raise ValueError(f"Frame size must be positive, got {(self._w, self._h)}")

        if self._fps <= 0:
            raise ValueError(f"fps must be positive, got {self._fps}")

        self._encoder_config = encoder_config
        self._input_pix_fmt = input_pix_fmt
        self._container_format = container_format
        self._open_options = dict(open_options or {})
        self._configure_stream = configure_stream

        self._container: Optional[av.container.OutputContainer] = None
        self._stream = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self.is_open:
            return

        cfg = self._encoder_config

        open_kwargs = dict(self._open_options)
        if self._container_format is not None:
            open_kwargs["format"] = self._container_format

        container = av.open(str(self._path), mode="w", **open_kwargs)

        try:
            rate = Fraction(self._fps).limit_denominator(1001)
            stream = container.add_stream(cfg.codec, rate=rate)

            stream.width = self._w
            stream.height = self._h
            stream.pix_fmt = cfg.pix_fmt
            stream.options = dict(cfg.options)

            for name, value in cfg.codec_context_attrs.items():
                setattr(stream.codec_context, name, value)

            if self._configure_stream is not None:
                self._configure_stream(stream)

        except Exception:
            container.close()
            raise

        self._container = container
        self._stream = stream
        self._frame_count = 0

    def close(self) -> None:
        if not self.is_open:
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
        Write one frame.

        The frame must match input_pix_fmt, not encoder_config.pix_fmt.

        Examples
        --------
        input_pix_fmt="bgr24":
            frame shape must be (H, W, 3), dtype uint8, BGR channel order.

        input_pix_fmt="rgb24":
            frame shape must be (H, W, 3), dtype uint8, RGB channel order.

        input_pix_fmt="gray":
            frame shape must be (H, W) or (H, W, 1), dtype uint8.
        """
        if self._container is None or self._stream is None:
            raise RuntimeError("PyAVStreamWriter is not open — call .open() first.")

        img = _as_ndarray(frame)
        img = _validate_and_normalize_frame(
            img,
            input_pix_fmt=self._input_pix_fmt,
            expected_size=(self._w, self._h),
        )

        av_frame = av.VideoFrame.from_ndarray(img, format=self._input_pix_fmt)
        av_frame.pts = self._frame_count
        self._frame_count += 1

        target_pix_fmt = self._stream.codec_context.pix_fmt
        if target_pix_fmt and av_frame.format.name != target_pix_fmt:
            av_frame = av_frame.reformat(format=target_pix_fmt)

        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    # ------------------------------------------------------------------
    # PyAV accessors
    # ------------------------------------------------------------------

    @property
    def container(self) -> av.container.OutputContainer:
        """
        The underlying PyAV output container.

        Available only while the writer is open.
        """
        if self._container is None:
            raise RuntimeError("Container is only available while writer is open.")
        return self._container

    @property
    def stream(self):
        """
        The underlying PyAV stream.

        Available only while the writer is open.
        """
        if self._stream is None:
            raise RuntimeError("Stream is only available while writer is open.")
        return self._stream

    @property
    def is_open(self) -> bool:
        return self._container is not None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def input_pix_fmt(self) -> str:
        return self._input_pix_fmt

    @property
    def encoder_config(self) -> EncoderConfig:
        return self._encoder_config
