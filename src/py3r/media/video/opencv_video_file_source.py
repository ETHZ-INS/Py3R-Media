import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class EndOfStreamError(Exception):
    """
    The video file has no more frames.

    When ``loop=False`` this is effectively fatal — the source is exhausted.
    When ``loop=True`` the reader will seek back to frame 0 and this should
    not normally be raised to callers.
    """


class ReadFailedError(Exception):
    """
    OpenCV returned ``ok=False`` on a mid-stream read (not at end-of-file).

    This is **retryable** — a single bad decode does not invalidate the
    capture object.  The caller may call ``read()`` again to attempt the
    next frame.
    """


class OpenCVVideoFileSource(VideoSource):
    """
    Pull-based video-file source backed by ``cv2.VideoCapture``.

    Unlike ``FFmpegVideoFileSource``, no subprocess is needed.  End-of-stream
    and decode failures are signalled via exceptions rather than ``None``
    returns so that upstream retry logic can distinguish recoverable hiccups
    from true end-of-file.

    Parameters
    ----------
    path:
        Path to the video file.
    grayscale:
        Convert frames to single-channel grayscale before returning them.
    loop:
        Seek back to frame 0 when EOS is reached and keep streaming.
    playback:
        Pace reads to the native frame-rate (suitable for display loops).
    """

    def __init__(
        self,
        path: Path,
        grayscale: bool = True,
        loop: bool = False,
        playback: bool = False,
    ) -> None:
        self._path = Path(path)
        self._grayscale = grayscale
        self._loop = loop
        self._playback = playback

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None
        self._size: Optional[Tuple[int, int]] = None
        self._num_frames: Optional[int] = None
        self._idx: int = 0
        self._t0: float = 0.0

        self._probe()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._cap = self._open_capture()
        self._idx = 0
        self._t0 = time.perf_counter()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return True
    def has_fps(self) -> bool: return bool(self._fps)
    def has_num_frames(self) -> bool: return not self._loop and self._num_frames is not None
    def is_seekable(self) -> bool: return True

    def get_size(self) -> Optional[Tuple[int, int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return 1 if self._grayscale else 3
    def get_num_frames(self) -> Optional[int]: return self._num_frames if not self._loop else None

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def seek(self, frame_index: int) -> None:
        if not self.is_open():
            raise RuntimeError("Cannot seek: source is not open")
        self._seek_file_position(frame_index)
        self._idx = max(0, int(frame_index))
        self._t0 = time.perf_counter() - (self._idx / (self._fps or 30.0))

    def _seek_file_position(self, frame_index: int) -> None:
        """Move the underlying capture to *frame_index* without touching ``_idx`` or ``_t0``."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, timeout: Optional[float] = None) -> VideoFrame:
        """
        Return the next ``VideoFrame`` from the file.

        Raises
        ------
        RuntimeError
            The source has not been opened.  This is fatal.
        EndOfStreamError
            No more frames are available (end of file).  When ``loop=True``
            this is handled internally by seeking back to frame 0; it will
            only propagate if the seek + re-read also fails.
        ReadFailedError
            OpenCV decoded a frame as invalid mid-stream.  This is retryable:
            skip the frame and call ``read()`` again.
        TimeoutError
            The read took longer than *timeout* seconds.  Retryable.
        """
        if not self.is_open():
            raise RuntimeError("Source is not open")

        t0 = time.perf_counter()

        ok, img = self._cap.read()  # type: ignore[union-attr]

        if timeout is not None and (time.perf_counter() - t0) > timeout:
            raise TimeoutError(f"Read exceeded timeout of {timeout:.3f}s")

        if not ok:
            # Distinguish true EOS from a mid-stream decode hiccup.
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)  # type: ignore[union-attr]
            at_eos = self._num_frames is None or pos >= self._num_frames

            if at_eos:
                if self._loop:
                    self._seek_file_position(0)  # reposition file only; _idx keeps counting
                    ok, img = self._cap.read()  # type: ignore[union-attr]
                    if not ok:
                        raise EndOfStreamError(
                            "End of stream even after loop seek — file may be empty or corrupt"
                        )
                else:
                    raise EndOfStreamError("End of video file reached")
            else:
                raise ReadFailedError(
                    f"OpenCV failed to decode frame at position {int(pos)}"
                )

        if self._grayscale and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Timestamp: prefer the container presentation time; fall back to
        # a synthetic clock derived from frame index + t0 (playback mode).
        ts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)  # type: ignore[union-attr]
        if ts_ms > 0:
            ts = ts_ms / 1000.0
        elif self._fps:
            ts = self._idx / self._fps
        else:
            ts = time.perf_counter() - self._t0

        if self._playback and self._fps:
            deadline = self._t0 + (self._idx / self._fps)
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        frame = VideoFrame(img, self._idx, ts)
        self._idx += 1
        return frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")
        return cap

    def _probe(self) -> None:
        cap = self._open_capture()
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._size = (w, h)

            fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(fps) if fps and fps > 0 else None

            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._num_frames = n if n > 0 else None
        finally:
            cap.release()

