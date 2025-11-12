from typing import Optional, Callable

import numpy as np

from py3r.media.types import HasImage
from py3r.media.video.ffmpeg_video_file_writer import FFmpegVideoFileWriter


class SegmentedVideoWriter:
    def __init__(
        self,
        video_writer_factory: Callable[[int], FFmpegVideoFileWriter],  # segment index -> writer
        segment_length_frames: int,
        segment_finished_callback: Optional[Callable[[int], None]] = None,
    ):
        self._video_writer_factory = video_writer_factory
        self._segment_length_frames = segment_length_frames
        self._segment_finished_callback = segment_finished_callback
        
        self._segment_index = 0
        self._segment_frame_index = 0
        self._current_writer: Optional[FFmpegVideoFileWriter] = None

        self._closed = False
        
    def write(self, frame: HasImage | np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed SegmentedVideoWriter")

        if self._current_writer is None or self._segment_frame_index >= self._segment_length_frames:
            self._close_segment()
            self._open_new_segment()
            self._segment_frame_index = 0
        
        img = frame.img if isinstance(frame, HasImage) else frame
        self._current_writer.write(img)
        self._segment_frame_index += 1

    def open(self) -> None:
        if self._closed:
            raise RuntimeError("SegmentedVideoWriter is already closed")
        if self._current_writer is not None:
            return
        self._open_new_segment()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_segment()

    @property
    def is_open(self) -> bool:
        return not self._closed and self._current_writer is not None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _open_new_segment(self):
        self._current_writer = self._video_writer_factory(self._segment_index)
        self._current_writer.open()
        self._segment_index += 1

    def _close_segment(self):
        if self._current_writer is None:
            return

        self._current_writer.close()

        if self._segment_frame_index > 0 and self._segment_finished_callback is not None:
            self._segment_finished_callback(self._segment_index - 1)
