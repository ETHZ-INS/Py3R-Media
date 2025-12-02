from threading import Lock
from typing import Union

import numpy as np
import reactivex as rx
from reactivex.disposable import Disposable

from py3r.media.types import HasImage
from py3r.media.video.ffmpeg_video_file_writer import FFmpegVideoFileWriter


# -------- resource bound to subscription via rx.using --------

class _VideoWriterResource(Disposable):
    """Owns writer teardown. Idempotent & thread-safe."""
    def __init__(self, writer: FFmpegVideoFileWriter):
        super().__init__()
        self._writer = writer
        self._closed = False
        self._lock = Lock()

    def dispose(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # noinspection PyBroadException
            try:
                self._writer.close()
            except Exception:
                pass

# -------- observer focused only on per-item work --------

class VideoWriterObserver(rx.Observer[Union[HasImage, np.ndarray]]):
    """
    Writes frames to a video file. Lifetime handled by rx.using.
    """
    def __init__(self, video_writer: FFmpegVideoFileWriter):
        super().__init__()
        self._video_writer = video_writer

    def using(self, upstream: rx.Observable[Union[HasImage, np.ndarray]]):
        def resource_factory():
            # Create the teardown resource now; open in observable_factory so we can
            # translate open() failures into an observable error.
            return _VideoWriterResource(self._video_writer)

        def observable_factory(_res):
            try:
                self._video_writer.open()  # open at subscription time
            except Exception as e:
                return rx.throw(e)  # propagate to observer.on_error
            return upstream

        return rx.using(resource_factory, observable_factory)

    def _on_next_core(self, frame: Union[HasImage, np.ndarray]):
        if not self._video_writer.is_open:
            return  # defensive: ignore late items after dispose/close
        try:
            self._video_writer.write(frame)
        except Exception as e:
            # Surface write failures to the stream; using() will close the writer.
            self.on_error(e)
