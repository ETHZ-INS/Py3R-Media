from __future__ import annotations

from typing import Callable
import threading

import cv2
import numpy as np
import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable, SerialDisposable
from reactivex.abc import SchedulerBase, ObserverBase


def opencv_imshow(
    window_name: str,
    *,
    scheduler: SchedulerBase,
    flags: int = cv2.WINDOW_AUTOSIZE,
    wait_ms: int = 1,
) -> Callable[[Observable[np.ndarray]], Observable[np.ndarray]]:
    def _operator(source: Observable[np.ndarray]) -> Observable[np.ndarray]:
        def _subscribe(
            observer: ObserverBase[np.ndarray],
            _subscribe_scheduler: SchedulerBase | None = None,
        ):
            lock = threading.Lock()
            disposed = False

            upstream = SerialDisposable()

            def create_window(*_):
                with lock:
                    if disposed:
                        return
                cv2.namedWindow(window_name, flags)

            scheduler.schedule(create_window)

            def on_next(img: np.ndarray) -> None:
                # Queue GUI work and return immediately.
                def show_image(*_):
                    with lock:
                        if disposed:
                            return

                    cv2.imshow(window_name, img)
                    cv2.waitKey(wait_ms)

                scheduler.schedule(show_image)
                observer.on_next(img)

            def on_error(err: Exception) -> None:
                observer.on_error(err)

            def on_completed() -> None:
                observer.on_completed()

            upstream.disposable = source.subscribe(
                on_next,
                on_error,
                on_completed,
            )

            def dispose() -> None:
                nonlocal disposed
                with lock:
                    if disposed:
                        return
                    disposed = True

                upstream.dispose()

                def destroy_window(*_):
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass

                scheduler.schedule(destroy_window)

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _operator
