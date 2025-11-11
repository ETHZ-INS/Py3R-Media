import threading

import cv2
import numpy as np

import reactivex as rx
from reactivex.disposable import Disposable

from py3r.media.types import HasImage


class CvWindow(Disposable):
    def __init__(self, name):
        super().__init__()
        self.name = name
        cv2.namedWindow(self.name)

    def dispose(self):
        cv2.destroyWindow(self.name)

class OpenCVImshowViewer(rx.Observer[HasImage | np.ndarray]):
    def __init__(self, window_name="Video"):
        super().__init__()
        self.window_name = window_name

    def on_next(self, item: HasImage | np.ndarray):
        frame = item if isinstance(item, np.ndarray) else item.img
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def using(self, upstream):
        def resource_factory():
            return CvWindow(self.window_name)

        def observable_factory(_):
            return upstream

        return rx.using(resource_factory, observable_factory)
