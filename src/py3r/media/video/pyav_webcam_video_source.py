import re
import subprocess
import threading
import time
from collections import deque
from typing import Optional, Tuple, List, Deque

import av

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class PyAVWebcamSource(VideoSource):
    """
    Windows / DirectShow webcam source via PyAV.

    Capture and decode are done by PyAV (FFmpeg bindings). Device enumeration
    helpers are kept subprocess-based for convenience, matching the original
    class behavior.

    Use either:
        - device_name="Integrated Camera"
    or:
        - device_index=0

    Notes
    -----
    - device_index is based on the order returned by ffmpeg dshow enumeration,
      just like your original class.
    - read(timeout=...) works via a background decode thread and a queue,
      rather than trying to interrupt a blocking decode call directly.
    - timestamps prefer frame/container timing when available; otherwise they
      fall back to time.perf_counter().
    """

    ffmpeg_executable = "ffmpeg"

    def __init__(
        self,
        device_name: Optional[str] = None,
        device_index: Optional[int] = None,
        grayscale: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        loglevel: str = "error",
        queue_size: int = 8,
    ):
        if device_name is None and device_index is None:
            device_index = 0
        if device_name is not None and device_index is not None:
            raise ValueError("Specify either device_name or device_index, not both.")

        self._device_name = device_name
        self._device_index = device_index
        self._device_number = 0  # duplicate-name selector for dshow

        self._grayscale = grayscale
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._loglevel = loglevel
        self._queue_size = max(1, int(queue_size))

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

        self._resolve_device()
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

    def has_timing(self) -> bool:
        return True

    def has_size(self) -> bool:
        return self._size is not None

    def has_fps(self) -> bool:
        return self._fps is not None

    def has_num_frames(self) -> bool:
        return False

    def is_seekable(self) -> bool:
        return False

    def get_size(self) -> Optional[Tuple[int, int]]:
        return self._size

    def get_fps(self) -> Optional[float]:
        return self._fps

    def get_num_channels(self) -> int:
        return self._channels

    def get_num_frames(self) -> Optional[int]:
        return None

    def seek(self, frame_index: int) -> None:
        # live source; no-op
        pass

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
    # Device enumeration helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_video_devices(cls) -> List[str]:
        """
        Returns video device names in the order ffmpeg lists them.

        Duplicate names are returned multiple times.
        """
        cmd = [
            str(cls.ffmpeg_executable),
            "-hide_banner",
            "-list_devices", "true",
            "-f", "dshow",
            "-i", "dummy",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        text = proc.stderr
        devices = []

        for line in text.splitlines():
            if not line.endswith("(video)"):
                continue
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append(m.group(1))

        return devices

    @classmethod
    def list_video_device_entries(cls) -> List[Tuple[int, str, int]]:
        """
        Returns [(global_index, device_name, video_device_number), ...].

        video_device_number is the dshow duplicate-name selector.
        """
        names = cls.list_video_devices()
        counts = {}
        out = []
        for i, name in enumerate(names):
            n = counts.get(name, 0)
            out.append((i, name, n))
            counts[name] = n + 1
        return out

    def _resolve_device(self) -> None:
        if self._device_name is not None:
            self._device_number = 0
            return

        entries = self.list_video_device_entries()
        if not entries:
            raise RuntimeError("No DirectShow video devices found")

        idx = int(self._device_index)
        if idx < 0 or idx >= len(entries):
            raise IndexError(
                f"device_index {idx} out of range; found {len(entries)} video device(s)"
            )

        _, name, devnum = entries[idx]
        self._device_name = name
        self._device_number = devnum

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def _probe(self) -> None:
        """
        Try to open the device briefly and inspect the video stream metadata.

        If width/height were explicitly requested, trust them as the expected
        output size. Otherwise use stream metadata when available.
        """
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
    # PyAV container / stream handling
    # ------------------------------------------------------------------

    def _build_dshow_options(self) -> dict:
        opts = {'rtbufsize': '500M'}

        if self._requested_fps is not None:
            # FFmpeg dshow private option
            opts["framerate"] = str(self._requested_fps)

        if self._requested_width is not None and self._requested_height is not None:
            # FFmpeg dshow private option
            opts["video_size"] = f"{self._requested_width}x{self._requested_height}"

        if self._device_number:
            # FFmpeg dshow private option for duplicate names
            opts["video_device_number"] = str(self._device_number)

        return opts

    def _open_container(self) -> av.container.InputContainer:
        if self._device_name is None:
            raise RuntimeError("Device name was not resolved")

        # PyAV's av.open accepts input format and options, which are passed through
        # to FFmpeg/libavformat.
        return av.open(
            file=f"video={self._device_name}",
            format="dshow",
            mode="r",
            options=self._build_dshow_options(),
        )

    @staticmethod
    def _select_video_stream(container: av.container.InputContainer):
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise RuntimeError("No video stream found from DirectShow device")
        stream = video_streams[0]

        # Let FFmpeg/PyAV use frame threading when available.
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
                    # Usually already (h, w), but normalize defensively.
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                else:
                    # Should already be (h, w, 3)
                    if arr.ndim != 3 or arr.shape[-1] != 3:
                        raise RuntimeError(
                            f"Unexpected color frame shape from PyAV: {arr.shape}"
                        )

                # Update size if it was unknown initially.
                if self._size is None:
                    if self._grayscale:
                        self._size = (int(arr.shape[1]), int(arr.shape[0]))
                    else:
                        self._size = (int(arr.shape[1]), int(arr.shape[0]))

                ts = self._frame_timestamp_seconds(frame)

                vf = VideoFrame(arr.copy(), self._idx, ts)
                self._idx += 1

                with self._cond:
                    # Keep only the newest frames if consumer is slower than source.
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
        """
        Prefer source/container-derived timing if present.

        PyAV frames may expose:
          - frame.time
          - frame.pts + frame.time_base

        Fallback is a monotonic clock sample at delivery time.
        """
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
