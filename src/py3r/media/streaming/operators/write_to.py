from __future__ import annotations

from typing import Callable, TypeVar, Protocol
import threading

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable
from reactivex.abc import ObserverBase, DisposableBase

TItem = TypeVar("TItem")


class IWriter(Protocol[TItem]):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def write(self, item: TItem) -> None: ...


def write_to(
    writer: IWriter[TItem],
) -> Callable[[Observable[TItem]], Observable[TItem]]:
    def _operator(source: Observable[TItem]) -> Observable[TItem]:
        def _subscribe(observer: ObserverBase[TItem], _=None) -> DisposableBase:
            lock = threading.RLock()
            closed = False

            def cleanup_once() -> None:
                nonlocal closed
                with lock:
                    if closed:
                        return
                    closed = True
                writer.close()

            writer.open()
            try:
                def on_next(item: TItem) -> None:
                    with lock:
                        if closed:
                            return
                        writer.write(item)
                        observer.on_next(item)

                def on_error(err: Exception) -> None:
                    cleanup_once()
                    observer.on_error(err)

                def on_completed() -> None:
                    cleanup_once()
                    observer.on_completed()

                upstream = source.subscribe(
                    on_next,
                    on_error,
                    on_completed,
                )
            except Exception:
                cleanup_once()
                raise

            def dispose() -> None:
                upstream.dispose()
                cleanup_once()

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _operator
