import time
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

from pypylon import genicam, pylon

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class GrabFailedError(BaseException):
    """The camera returned a result whose GrabSucceeded() flag is False."""


class GrabTimeoutError(BaseException):
    """RetrieveResult returned without a frame within the given timeout."""


class PylonCameraSource(VideoSource):
    """
    Pull-based Basler camera source via pypylon.

    Parameters
    ----------
    serial : str
        Camera serial number used to locate the device via TlFactory.
    config_file : Path | None
        PFS feature-persistence file to load after opening.  Errors are
        swallowed so that emulated cameras or mismatched configs don't crash
        at startup.
    camera_factory : CameraFactory | None
        If provided, replaces the real ``TlFactory`` device enumeration.
        Called with the serial number and must return an already-opened
        camera object::

            camera_factory(serial) -> camera

        Use this in unit tests to inject a fake camera without needing
        physical hardware or a pypylon installation.
    """

    class CameraFactory(Protocol):
        def __call__(self, serial: str) -> Any: ...

    def __init__(
        self,
        serial: str,
        config_file: Optional[Path] = None,
        *,
        camera_factory: Optional[CameraFactory] = None,
    ):
        self._serial = serial
        self._config_file = config_file
        self._camera_factory = camera_factory
        self._cam = None
        self._idx = 0
        self._size: Optional[Tuple[int, int]] = None
        self._fps: Optional[float] = None
        self._gray = False
        self._tick_frequency: int = 1_000_000_000  # default: assume ns until probed

        self._probe()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._cam = self._open_camera()
        self._configure_camera(self._cam)
        self._cam.MaxNumBuffer = 30
        self._cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        self._idx = 0

    def close(self) -> None:
        if self._cam:
            self._cam.StopGrabbing()
            self._cam.Close()
        self._cam = None

    def is_open(self) -> bool:
        return self._cam is not None and self._cam.IsOpen()

    def has_timing(self) -> bool: return True
    def has_size(self) -> bool: return self._size is not None
    def has_fps(self) -> bool: return self._fps is not None
    def has_num_frames(self) -> bool: return False
    def is_seekable(self) -> bool: return False

    def get_size(self) -> Optional[Tuple[int, int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return 1 if self._gray else 3
    def get_num_frames(self) -> Optional[int]: return None

    def set_playback_rate(self, mode: str) -> None: pass
    def seek(self, frame_index: int) -> None: pass

    def read(self, timeout: Optional[float] = None) -> VideoFrame:
        """
        Read the next frame from the camera.

        Raises
        ------
        GrabTimeoutError
            No frame within *timeout* seconds — retryable.
        GrabFailedError
            SDK returned GrabSucceeded()==False (dropped frame) — retryable.
        RuntimeError
            Camera not open or stopped grabbing — fatal.
        """
        if not self._cam or not self._cam.IsGrabbing():
            raise RuntimeError("Camera is not open or has stopped grabbing")

        grab_timeout_ms = 500 if timeout is None else int(timeout * 1000)

        # TimeoutHandling_Return gives back None on timeout instead of raising,
        # so we can translate it into our typed exception hierarchy cleanly.
        result = self._cam.RetrieveResult(grab_timeout_ms, pylon.TimeoutHandling_Return)

        # TimeoutHandling_Return yields an empty (invalid) GrabResult on timeout,
        # not None — use __bool__ / IsValid() rather than identity check.
        if not result:
            raise GrabTimeoutError(f"No frame received within {timeout or 0.5:.3f}s")

        if not result.GrabSucceeded():
            err_code = result.GetErrorCode()
            err_desc = result.GetErrorDescription()
            result.Release()
            raise GrabFailedError(f"Grab failed (code={err_code:#010x}): {err_desc}")

        img = result.Array.copy()  # copy before Release

        # BlockID is the hardware frame counter — gaps are immediately visible
        # to downstream pipelines.  Fall back to our own counter if unavailable.
        block_id = getattr(result, "BlockID", None)
        frame_index = int(block_id) if block_id is not None else self._idx

        # Device timestamp in hardware ticks; divide by tick frequency for seconds.
        ts_device_ticks = getattr(result, "TimeStamp", None)
        ts = (ts_device_ticks / self._tick_frequency) if ts_device_ticks else time.perf_counter()

        result.Release()

        self._idx += 1
        return VideoFrame(img, frame_index, ts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_camera(self):
        if self._camera_factory is not None:
            return self._camera_factory(self._serial)

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found")
        dev = next((d for d in devices if d.GetSerialNumber() == self._serial), None)
        if dev is None:
            raise RuntimeError(f"Basler camera with serial '{self._serial}' not found")
        cam = pylon.InstantCamera(tl_factory.CreateDevice(dev))
        cam.Open()
        return cam

    def _configure_camera(self, camera) -> None:
        if self._config_file is not None:
            try:
                pylon.FeaturePersistence.Load(
                    str(self._config_file), camera.GetNodeMap(), True
                )
            except Exception:
                # Config files made for a real GigE camera will fail on the
                # emulator (and vice-versa) because device-specific nodes don't
                # exist. Swallow so the camera is still usable.
                pass

        # GigE packet size — silently ignored on USB / emulated cameras.
        try:
            camera.GevSCPSPacketSize.SetValue(1500)
        except (genicam.LogicalErrorException, AttributeError):
            pass

    def _probe(self) -> None:
        try:
            cam = self._open_camera()
            try:
                self._configure_camera(cam)

                try:
                    self._tick_frequency = int(cam.GevTimestampTickFrequency.GetValue())
                except (genicam.LogicalErrorException, AttributeError):
                    pass  # USB / emulated cameras — keep default 1_000_000_000

                self._size = (int(cam.Width.GetValue()), int(cam.Height.GetValue()))

                try:
                    fps = cam.AcquisitionFrameRateAbs.GetValue()
                    self._fps = float(fps) if fps else 30.0
                except (genicam.LogicalErrorException, AttributeError):
                    self._fps = 30.0

                try:
                    self._gray = cam.PixelFormat.GetValue() == "Mono8"
                except (genicam.LogicalErrorException, AttributeError):
                    self._gray = False
            finally:
                cam.Close()
        except Exception:
            # Soft-fail: caller can still open() later.
            pass
