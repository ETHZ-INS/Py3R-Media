from dataclasses import dataclass, replace
from typing import Tuple, runtime_checkable, Protocol, Union

import numpy as np


# ---------------------------------------------------------------------------
# Read error hierarchy
# ---------------------------------------------------------------------------

class ReadError(Exception):
    """Base class for errors raised by IReader.read()."""


class FatalReadError(ReadError):
    """
    An unrecoverable read failure.  ``reader_observable`` will forward this
    immediately as ``on_error`` without consulting ``max_consecutive_errors``.

    All other exceptions (including plain ``ReadError`` subclasses) are treated
    as retryable by default — the source will be retried up to
    ``max_consecutive_errors`` times before the stream is terminated.

    Use this only when retrying is known to be pointless or harmful (e.g. the
    camera has been physically disconnected and the source has already set an
    internal flag so that every subsequent ``read()`` would also fail
    immediately).  In most cases it is better to let the consecutive-error
    threshold handle termination naturally.
    """


class GrabFailedError(ReadError):
    """The camera returned a result whose GrabSucceeded() flag is False."""


class GrabTimeoutError(ReadError):
    """RetrieveResult returned without a frame within the given timeout."""


@runtime_checkable
class HasImage(Protocol):
    @property
    def img(self) -> np.ndarray: ...

    @property
    def is_color(self) -> bool:
        return self.img.ndim == 3 and self.img.shape[2] == 3

@runtime_checkable
class HasImageMeta(Protocol):
    @property
    def size(self) -> Tuple[int, int]: ...  # width, height

@runtime_checkable
class HasFrameMeta(HasImageMeta, Protocol):
    @property
    def frame_index(self) -> int: ...
    @property
    def timestamp(self) -> float: ...

@dataclass(frozen=True, slots=True)
class ImageMeta(HasImageMeta):
    # noinspection PyProtocol
    size: Tuple[int, int]

    @classmethod
    def from_meta(cls, o: HasImageMeta) -> "ImageMeta":
        return cls(o.size)

@dataclass(frozen=True, slots=True)
class FrameMeta(HasFrameMeta):
    # noinspection PyProtocol
    size: Tuple[int, int]
    # noinspection PyProtocol
    frame_index: int = 0
    # noinspection PyProtocol
    timestamp: float = 0.0

    @classmethod
    def from_meta(cls, o: HasFrameMeta) -> "FrameMeta":
        return cls(o.size, o.frame_index, o.timestamp)

@dataclass(frozen=True, slots=True)
class Image(HasImage, HasImageMeta):
    # noinspection PyProtocol
    img: np.ndarray

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h

@dataclass(frozen=True, slots=True)
class VideoFrame(HasImage, HasFrameMeta):
    # noinspection PyProtocol
    img: np.ndarray
    # noinspection PyProtocol
    frame_index: int = 0
    # noinspection PyProtocol
    timestamp: float = 0.0

    @classmethod
    def from_parts(cls, img: Union[HasImage, np.ndarray], meta: HasFrameMeta) -> "VideoFrame":
        return cls(
            img=img if isinstance(img, np.ndarray) else img.img,
           frame_index=meta.frame_index,
           timestamp=meta.timestamp,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[Union[HasImage, np.ndarray], HasFrameMeta]) -> "VideoFrame":
        return cls.from_parts(*pair)

    def with_image(self, img: Union[HasImage, np.ndarray]) -> "VideoFrame":
        return replace(self, img=img if isinstance(img, np.ndarray) else img.img)

    def with_meta(self, meta: HasFrameMeta) -> "VideoFrame":
        return replace(
            self,
            frame_index=meta.frame_index,
            timestamp=meta.timestamp,
        )

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h
