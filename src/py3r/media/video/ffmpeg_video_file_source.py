import subprocess, json, numpy as np, time, sys
from pathlib import Path
from typing import Optional, Tuple

from py3r.media.types import VideoFrame


class FFmpegVideoFileSource:
    def __init__(self, path: Path, loop: bool = False, grayscale: bool = True, loglevel: str = "error", pipe_size_bytes: int | None = None):
        self._path = Path(path)
        self._loop = loop
        self._grayscale = grayscale
        self._loglevel = loglevel
        self._proc: Optional[subprocess.Popen] = None
        self._fps: Optional[float] = None
        self._size: Optional[Tuple[int,int]] = None
        self._channels: int = 1 if grayscale else 3
        self._num_frames: Optional[int] = None

        self._mode: str = "original_speed"
        self._idx: int = 0
        self._t0: float = 0.0

        # preallocated IO buffers
        self._frame_nbytes: int = 0
        self._frame_bytes: Optional[bytearray] = None
        self._mv = None  # memoryview(self._frame_bytes)

        self._pipe_size_bytes = pipe_size_bytes  # Linux only; ignored elsewhere

        self._probe()

    # --- protocol-ish bits (same as before) ---
    def open(self) -> None:
        self._start_proc()
        self._idx = 0
        self._t0 = time.perf_counter()

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                self._proc.kill()
        self._proc = None

    def is_open(self) -> bool: return self._proc is not None and self._proc.poll() is None

    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return True
    def has_fps(self) -> bool: return True
    def has_num_frames(self) -> bool: return not self._loop
    def is_seekable(self) -> bool: return True

    def get_size(self) -> Optional[Tuple[int,int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return self._channels
    def get_num_frames(self) -> Optional[int]: return self._num_frames

    def set_playback_rate(self, mode: str) -> None:
        assert mode in {"original_speed", "max_speed"}
        self._mode = mode

    def seek(self, frame_index: int) -> None:
        # Simple robust seek: restart ffmpeg and discard N frames
        self.close()
        self._start_proc()
        drop = max(0, int(frame_index))
        self._discard_n(drop)
        self._idx = drop
        self._t0 = time.perf_counter() - (self._idx / (self._fps or 30.0))

    def enable_grayscale(self, gray: bool) -> None:
        if gray == self._grayscale:
            return
        self._grayscale = gray
        self._channels = 1 if gray else 3
        self.close()
        self._start_proc()

    # --- main read with preallocated buffer + readinto ---
    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        if not self.is_open():
            return None
        stdout = self._proc.stdout  # type: ignore
        if stdout is None:
            return None

        ok = self._read_exact_into(stdout, timeout=timeout)
        if not ok:
            if self._loop:
                self.seek(0)
                ok = self._read_exact_into(stdout, timeout=timeout)
                if not ok:
                    return None
            else:
                return None  # EOS or timeout

        # Zero-copy view from the reused buffer
        arr = np.frombuffer(self._frame_bytes, dtype=np.uint8, count=self._frame_nbytes).copy()
        w, h = self._size
        if self._channels == 1:
            img = arr.reshape((h, w))
        else:
            img = arr.reshape((h, w, 3))  # BGR

        ts = None
        if self._mode == "original_speed" and self._fps:
            ts = self._idx / self._fps
            # Pace to original speed
            delay = (self._t0 + ts) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        frame = VideoFrame(img, self._idx, ts)
        self._idx += 1
        return frame

    # --- internals ---
    def _probe(self):
        out = subprocess.check_output([
            "ffprobe","-v","error","-select_streams","v:0","-show_streams","-of","json",str(self._path)
        ])
        info = json.loads(out.decode("utf-8"))
        v = info["streams"][0]
        w, h = int(v["width"]), int(v["height"])
        self._size = (w, h)
        fr = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
        self._fps = self._parse_fps(fr) or 30.0

        self._frame_nbytes = w * h * (1 if self._grayscale else 3)
        self._frame_bytes = bytearray(self._frame_nbytes)
        self._mv = memoryview(self._frame_bytes)

    def _start_proc(self):
        pix_fmt = "gray" if self._grayscale else "bgr24"
        # (Re)ensure buffer sizes are correct
        w, h = self._size
        self._frame_nbytes = w * h * (1 if self._grayscale else 3)
        if self._frame_bytes is None or len(self._frame_bytes) != self._frame_nbytes:
            self._frame_bytes = bytearray(self._frame_nbytes)
            self._mv = memoryview(self._frame_bytes)

        cmd = [
            "ffmpeg","-hide_banner","-loglevel", self._loglevel,
            "-nostdin",              # avoid blocking on stdin in some envs
            "-i", str(self._path),
            "-an","-sn","-dn",
            "-pix_fmt", pix_fmt,
            "-f","rawvideo","pipe:1"
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0  # let OS/ffmpeg batch the writes
        )

        # Optional: enlarge pipe buffer on Linux to reduce writer stalls on big frames
        if self._pipe_size_bytes and sys.platform.startswith("linux"):
            try:
                import fcntl
                fcntl.fcntl(self._proc.stdout.fileno(), 1031, self._pipe_size_bytes)  # F_SETPIPE_SZ = 1031
            except Exception:
                pass

    @staticmethod
    def _parse_fps(frac: str) -> Optional[float]:
        if "/" in frac:
            n, d = frac.split("/")
            n, d = float(n), float(d)
            return n/d if d else None
        try:
            return float(frac)
        except Exception:
            return None

    def _read_exact_into(self, stdout, timeout: Optional[float]) -> bool:
        """
        Fill the preallocated buffer exactly once per frame, using readinto().
        Returns False on EOS or timeout.
        """
        remaining = self._frame_nbytes
        offset = 0
        start = time.perf_counter()
        while remaining > 0:
            # read directly into the slice of the memoryview
            n = stdout.readinto(self._mv[offset:offset+remaining])
            if not n:
                return False  # EOS
            remaining -= n
            offset += n
            if timeout is not None and (time.perf_counter() - start) > timeout:
                raise TimeoutError("Timeout reading frame")
        return True

    def _discard_n(self, n: int):
        if n <= 0:
            return
        stdout = self._proc.stdout  # type: ignore
        for _ in range(n):
            if not self._read_exact_into(stdout, timeout=2.0):
                break
