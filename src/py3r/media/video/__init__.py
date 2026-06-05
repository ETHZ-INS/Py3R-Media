from typing import Optional, Tuple, Protocol, runtime_checkable, Union

import numpy as np

from py3r.media.types import VideoFrame, HasImage


@runtime_checkable
class VideoSource(Protocol):
    """A pull-based, minimal interface. Keep it small & capability-based."""
    # --- lifecycle
    def open(self) -> None: ...
    def close(self) -> None: ...
    def is_open(self) -> bool: ...

    # --- capabilities/probing (callable pre-open; stable after open)
    def has_timing(self) -> bool: ...

    def has_size(self) -> bool: ...  # Has a known frame size
    def has_fps(self) -> bool: ...  # Has a known frame rate
    def has_num_frames(self) -> bool: ...  # Has a known number of frames
    def is_seekable(self) -> bool: ...  # Can seek to a specific frame

    def get_size(self) -> Optional[Tuple[int, int]]: ...  # (width, height) or None if unknown/variable
    def get_fps(self) -> Optional[float]: ...  # fps or None if unknown/variable
    def get_num_channels(self) -> int: ...  # Usually 1 or 3, but could be more for special devices
    def get_num_frames(self) -> Optional[int]: ...  # Number of frames or None if live/unknown

    def seek(self, frame_index: int) -> None: ...

    # --- acquisition
    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        """
        Return next Frame, or None on end-of-stream (EOS).
        Should block until a frame is available (or timeout).
        """
        ...


@runtime_checkable
class VideoWriter(Protocol):
    """A minimal interface for writing video frames to a sink."""
    def open(self) -> None: ...
    def close(self) -> None: ...
    def is_open(self) -> bool: ...

    def write(self, frame: Union[HasImage, np.ndarray]) -> None: ...
