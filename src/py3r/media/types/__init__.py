from dataclasses import dataclass
from typing import Tuple, runtime_checkable, Protocol

import numpy as np


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
class ImageMeta(HasImageMeta):  # implements HasImage & HasSize
    _size: Tuple[int, int]

    @classmethod
    def from_meta(cls, o: HasImageMeta) -> "ImageMeta":
        return cls(o.size)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

@dataclass(frozen=True, slots=True)
class FrameMeta(HasFrameMeta):  # implements HasImage & HasSize
    _size: Tuple[int, int]
    _frame_index: int = 0
    _timestamp: float = 0.0

    @classmethod
    def from_meta(cls, o: HasFrameMeta) -> "FrameMeta":
        return cls(o.size, o.frame_index, o.timestamp)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp(self) -> float:
        return self._timestamp

@dataclass(frozen=True, slots=True)
class Image(HasImage, HasImageMeta):  # implements HasImage & HasSize
    _img: np.ndarray

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h

@dataclass(frozen=True, slots=True)
class VideoFrame(HasImage, HasFrameMeta):  # inherits only from Image concretely
    _img: np.ndarray
    _frame_index: int = 0
    _timestamp: float = 0.0

    @classmethod
    def from_parts(cls, img: HasImage | np.ndarray, meta: HasFrameMeta) -> "VideoFrame":
        return cls(
            _img=img if isinstance(img, np.ndarray) else img.img,
            _frame_index=meta.frame_index,
            _timestamp=meta.timestamp,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[HasImage | np.ndarray, HasFrameMeta]) -> "VideoFrame":
        return cls.from_parts(*pair)

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp(self) -> float:
        return self._timestamp
