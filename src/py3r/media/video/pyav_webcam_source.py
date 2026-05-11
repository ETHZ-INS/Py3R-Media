import threading
import time
from collections import deque
from typing import Any, Deque, Optional, Protocol, Tuple

import av

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class PyAVWebcamSource(VideoSource):
    """
    Pull-based webcam / capture-device source via PyAV (libav*).

    No ffmpeg executable is required — all capture and decode happens through
    the PyAV bindings directly.

    Parameters
    ----------
    device_name : str
        Platform-specific device identifier passed to libavformat.
        Required — pass a dummy string (e.g. ``"test"``) when supplying a
        ``container_factory`` in tests.

        * **Windows DirectShow** — the display name shown in Device Manager,
          e.g. ``"Integrated Camera"`` or ``"USB Video Device"``.
        * **Linux V4L2** — the device node, e.g. ``"/dev/video0"``.
        * **macOS AVFoundation** — the device index as a string, e.g. ``"0"``.
    input_format : str | None
        libavformat input format name.  Defaults to the platform default:
        ``"dshow"`` on Windows, ``"v4l2"`` on Linux, ``"avfoundation"`` on
        macOS.  Pass explicitly to override (e.g. ``"mjpeg"`` for some
        USB cameras).
    grayscale : bool
        Convert frames to single-channel grayscale before returning them.
    width, height : int | None
        Request a specific capture resolution.  The device may ignore or
        round this.
    fps : float | None
        Request a specific frame-rate.
    queue_size : int
        Maximum number of decoded frames kept in the internal queue.  When
        the producer outpaces the consumer, the *oldest* queued frames are
        silently dropped to make room.
    container_factory : ContainerFactory | None
        If provided, replaces ``av.open(...)`` for both the probe and the
        live capture session.  Called with the same arguments as the internal
        ``av.open`` call::

            container_factory(file, format, mode, options) -> container

        This lets tests (a) verify that the source constructs the right
        device identifier, format string, and capture-option dict, and
        (b) return a fake container to drive the rest of the pipeline.

    Notes
    -----
    * ``read(timeout=...)`` is non-blocking on the decode side: a background
      thread feeds frames into a bounded deque and ``read()`` pops from it.
    * Timestamps prefer container / frame PTS when available and fall back
      to ``time.perf_counter()`` at frame delivery time.
    """

    # Mirrors the av.open arguments used by _open_container.
    class ContainerFactory(Protocol):
        def __call__(
            self,
            file: str,
            *,
            format: str,
            mode: str,
            options: dict,
        ) -> Any: ...

    def __init__(
        self,
        device_name: str,
        *,
        input_format: Optional[str] = None,
        grayscale: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        queue_size: int = 8,
        container_factory: Optional[ContainerFactory] = None,
    ):

        self._device_name = device_name
        self._input_format = input_format or self._default_input_format()

        self._grayscale = grayscale
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._queue_size = max(1, int(queue_size))
        self._container_factory = container_factory

        self._idx = 0
        self._size: Optional[Tuple[int, int]] = None
        self._fps: Optional[float] = fps
        self._channels = 1 if grayscale else 3

        self._container: Optional[av.container.InputContainer] = None
        self._stream = None

        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._cond = threading.Condition()
        self._queue: Deque[VideoFrame] = deque()
        self._reader_error: Optional[BaseException] = None
        self._reader_eof = False

        self._probe()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self.is_open():
            return

        self._idx = 0
        self._reader_error = None
        self._reader_eof = False
        self._queue.clear()
        self._stop_event.clear()

        container = self._open_container()
        stream = self._select_video_stream(container)

        self._container = container
        self._stream = stream

        # Re-probe from the opened stream in case metadata is now available.
        self._update_size_and_fps_from_stream(stream)

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="pyav-webcam-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def close(self) -> None:
        self._stop_event.set()

        container = self._container
        self._container = None
        self._stream = None

        # Closing the container should unblock decode/demux on most paths.
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

        t = self._reader_thread
        self._reader_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

        with self._cond:
            self._queue.clear()
            self._reader_eof = True
            self._cond.notify_all()

    def is_open(self) -> bool:
        t = self._reader_thread
        return t is not None and t.is_alive() and not self._stop_event.is_set()

    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return self._size is not None
    def has_fps(self) -> bool: return self._fps is not None
    def has_num_frames(self) -> bool: return False
    def is_seekable(self) -> bool: return False

    def get_size(self) -> Optional[Tuple[int, int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return self._channels
    def get_num_frames(self) -> Optional[int]: return None

    def seek(self, frame_index: int) -> None:
        pass  # live source; no-op

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        deadline = None if timeout is None else (time.perf_counter() + timeout)

        with self._cond:
            while True:
                if self._queue:
                    return self._queue.popleft()

                if self._reader_error is not None:
                    raise RuntimeError("PyAV webcam reader failed") from self._reader_error

                if self._reader_eof:
                    return None

                if deadline is None:
                    self._cond.wait()
                    continue

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Timeout reading frame")

                self._cond.wait(timeout=remaining)

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def _probe(self) -> None:
        if self._requested_width is not None and self._requested_height is not None:
            self._size = (self._requested_width, self._requested_height)
        else:
            self._size = None

        self._fps = self._requested_fps

        try:
            container = self._open_container()
            try:
                stream = self._select_video_stream(container)
                self._update_size_and_fps_from_stream(stream)

                # If size is still unknown, decode one frame to force discovery.
                if self._size is None:
                    for frame in container.decode(stream):
                        self._size = (int(frame.width), int(frame.height))
                        break
            finally:
                container.close()
        except Exception:
            # Keep probing soft: caller can still open later and discover size there.
            pass

    def _update_size_and_fps_from_stream(self, stream) -> None:
        if self._size is None:
            w = getattr(stream.codec_context, "width", 0) or getattr(stream, "width", 0)
            h = getattr(stream.codec_context, "height", 0) or getattr(stream, "height", 0)
            if w and h:
                self._size = (int(w), int(h))

        if self._fps is None:
            rate = None
            for attr in ("average_rate", "base_rate", "guessed_rate"):
                try:
                    rate = getattr(stream, attr, None)
                except Exception:
                    rate = None
                if rate:
                    break
            if rate:
                try:
                    self._fps = float(rate)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Container / stream handling
    # ------------------------------------------------------------------

    def _open_container(self) -> av.container.InputContainer:
        container_factory = self._container_factory or av.open

        # DirectShow requires "video=<name>" to select the video device;
        # V4L2 and AVFoundation use the path/index directly.
        if self._input_format == "dshow":
            file = f"video={self._device_name}"
        else:
            file = self._device_name
        fmt = self._input_format
        mode = "r"
        options = self._build_capture_options()


        return container_factory(file=file, format=fmt, mode=mode, options=options)

    def _build_capture_options(self) -> dict:
        opts: dict = {}

        if self._requested_fps is not None:
            opts["framerate"] = str(self._requested_fps)

        if self._requested_width is not None and self._requested_height is not None:
            opts["video_size"] = f"{self._requested_width}x{self._requested_height}"

        # DirectShow: enlarge the real-time capture buffer to reduce drops
        if self._input_format == "dshow":
            opts["rtbufsize"] = "500M"

        return opts

    @staticmethod
    def _default_input_format() -> str:
        import sys
        if sys.platform == "win32":
            return "dshow"
        if sys.platform == "darwin":
            return "avfoundation"
        return "v4l2"

    @staticmethod
    def _select_video_stream(container: av.container.InputContainer):
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise RuntimeError("No video stream found in container")
        stream = video_streams[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        return stream

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        container = self._container
        stream = self._stream

        try:
            if container is None or stream is None:
                raise RuntimeError("Reader started without an open container")

            target_format = "gray" if self._grayscale else "bgr24"

            for frame in container.decode(stream):
                if self._stop_event.is_set():
                    break

                arr = frame.to_ndarray(format=target_format)

                if self._grayscale:
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                else:
                    if arr.ndim != 3 or arr.shape[-1] != 3:
                        raise RuntimeError(
                            f"Unexpected color frame shape from PyAV: {arr.shape}"
                        )

                if self._size is None:
                    self._size = (int(arr.shape[1]), int(arr.shape[0]))

                ts = self._frame_timestamp_seconds(frame)

                vf = VideoFrame(arr.copy(), self._idx, ts)
                self._idx += 1

                with self._cond:
                    while len(self._queue) >= self._queue_size:
                        self._queue.popleft()
                    self._queue.append(vf)
                    self._cond.notify()

        except Exception as exc:
            with self._cond:
                self._reader_error = exc
                self._reader_eof = True
                self._cond.notify_all()
            return

        with self._cond:
            self._reader_eof = True
            self._cond.notify_all()

    @staticmethod
    def _frame_timestamp_seconds(frame) -> float:
        try:
            t = getattr(frame, "time", None)
            if t is not None:
                return float(t)
        except Exception:
            pass

        try:
            pts = getattr(frame, "pts", None)
            tb = getattr(frame, "time_base", None)
            if pts is not None and tb is not None:
                return float(pts * tb)
        except Exception:
            pass

        return time.perf_counter()
