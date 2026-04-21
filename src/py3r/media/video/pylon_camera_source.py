import time
from pathlib import Path
from typing import Optional, Tuple

from pypylon import pylon

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


class GrabFailedError(BaseException):
    """The camera returned a result whose GrabSucceeded() flag is False."""


class GrabTimeoutError(BaseException):
    """RetrieveResult returned without a frame within the given timeout."""


class PylonCameraSource(VideoSource):
    def __init__(self, serial: str, config_file: Optional[Path] = None):
        self._serial = serial
        self._config_file = config_file
        self._cam = None
        self._idx = 0
        self._size = None
        self._fps = None
        self._gray = False

        self._probe()

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

    def is_open(self) -> bool: return self._cam is not None and self._cam.IsOpen()
    def has_timing(self) -> bool: return True  # device timestamp
    def has_size(self) -> bool: return True
    def has_fps(self) -> bool: return bool(self._fps)
    def has_num_frames(self) -> bool: return False
    def is_seekable(self) -> bool: return False

    def get_size(self) -> Optional[Tuple[int,int]]: return self._size
    def get_fps(self) -> Optional[float]: return self._fps
    def get_num_channels(self) -> int: return 1 if self._gray else 3
    def get_num_frames(self) -> Optional[int]: return None

    def set_playback_rate(self, mode: str) -> None: pass
    def seek(self, frame_index: int) -> None: pass

    def read(self, timeout: Optional[float] = None) -> VideoFrame:
        """
        Read the next frame from the camera.

        Returns a VideoFrame on success.

        Raises
        ------
        GrabTimeoutError
            The camera did not deliver a frame within *timeout* seconds.
            This is **retryable** — the camera is still running.
        GrabFailedError
            The SDK returned a result but ``GrabSucceeded()`` was False (e.g.
            incomplete / dropped frame due to CPU or network load).
            This is also **retryable** — individual dropped frames are normal.
        RuntimeError
            The camera is not open or has stopped grabbing.  This is fatal.
        """
        if not self._cam or not self._cam.IsGrabbing():
            raise RuntimeError("Camera is not open or has stopped grabbing")

        grab_timeout_ms = int((timeout or 0.5) * 1000)

        # Use TimeoutHandling_Return so the SDK gives us back a None/invalid
        # result on timeout rather than raising its own exception, which lets
        # us translate it into our typed hierarchy cleanly.
        result = self._cam.RetrieveResult(grab_timeout_ms, pylon.TimeoutHandling_Return)

        if result is None:
            raise GrabTimeoutError(
                f"No frame received within {timeout or 0.5:.3f}s"
            )

        if not result.GrabSucceeded():
            err_code = result.GetErrorCode()
            err_desc = result.GetErrorDescription()
            result.Release()
            raise GrabFailedError(
                f"Grab failed (code={err_code:#010x}): {err_desc}"
            )

        img = result.Array  # numpy view — copy before Release
        img = img.copy()
        ts_device_ns = getattr(result, "TimeStamp", None)
        ts = (ts_device_ns / 1e9) if ts_device_ns else time.perf_counter()
        result.Release()

        f = VideoFrame(img, self._idx, ts)
        self._idx += 1
        return f

    def _open_camera(self) -> pylon.InstantCamera:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found")
        dev = next((dev for dev in devices if dev.GetSerialNumber() == self._serial), None)
        if dev is None:
            raise RuntimeError(f"Basler camera with serial '{self._serial}' not found")
        cam = pylon.InstantCamera(tl_factory.CreateDevice(dev))
        cam.Open()
        return cam

    def _configure_camera(self, camera: pylon.InstantCamera):
        jumbo_frames = False

        if self._config_file is not None:
            pylon.FeaturePersistence.Load(str(self._config_file), camera.GetNodeMap(), True)
        elif self._serial.startswith("0815-"):
            camera.Width.SetValue(1280)
            camera.Height.SetValue(1024)
            camera.AcquisitionFrameRateAbs.SetValue(30.0)

        frame_size = 9000 if jumbo_frames else 1500
        if hasattr(camera, "GevSCPSPacketSize"):
            camera.GevSCPSPacketSize.SetValue(frame_size)

    def _probe(self):
        cam = self._open_camera()
        self._configure_camera(cam)

        width = cam.Width.GetValue()
        height = cam.Height.GetValue()
        self._size = (width, height)

        if hasattr(cam, "AcquisitionFrameRateAbs"):
            self._fps = cam.AcquisitionFrameRateAbs.GetValue() or 30.0
        else:
            self._fps = 30.0

        if cam.PixelFormat == "Mono8":
            self._gray = True
        else:
            self._gray = False
