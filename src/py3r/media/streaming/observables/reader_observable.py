from __future__ import annotations

from typing import Protocol, TypeVar
import threading

import reactivex as rx
from reactivex import Observable
from reactivex.abc import ObserverBase
from reactivex.disposable import Disposable


TItem = TypeVar("TItem")


class IReader(Protocol[TItem]):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def read(self, timeout: float) -> TItem: ...


def reader_observable(reader: IReader[TItem], read_timeout_seconds: float = 0.5) -> Observable[TItem]:
    """
    Turn any blocking reader object into an Observable sequence. At subscription time,
    starts a background thread to read from the reader one item at a time.

    :param reader: The reader object to read from. Must implement open(), close(), and read(timeout) methods.
    :param read_timeout_seconds: The timeout to use for each read operation, in seconds. This prevents the reader from blocking indefinitely and allows for responsive cancellation.
    :return:
    """
    def _subscribe(observer: ObserverBase[TItem], _=None) -> Disposable:
        stop = threading.Event()
        done = threading.Event()

        def run():
            try:
                reader.open()

                while not stop.is_set():
                    try:
                        item = reader.read(read_timeout_seconds)
                    except Exception as ex:
                        if not stop.is_set():
                            observer.on_error(ex)
                        return

                    if stop.is_set():
                        return

                    observer.on_next(item)

                # No on_completed on disposal/cancellation.

            except Exception as ex:
                if not stop.is_set():
                    observer.on_error(ex)
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
                done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        def dispose():
            stop.set()
            done.wait(timeout=5.0)

        return Disposable(dispose)

    return rx.create(_subscribe)
