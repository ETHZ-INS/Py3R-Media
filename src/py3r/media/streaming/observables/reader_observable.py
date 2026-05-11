from __future__ import annotations

import logging
from typing import Protocol, TypeVar, Optional
import threading

import reactivex as rx
from reactivex import Observable
from reactivex.abc import ObserverBase
from reactivex.disposable import Disposable

from py3r.media.types import FatalReadError

log = logging.getLogger(__name__)

TItem = TypeVar("TItem")


class IReader(Protocol[TItem]):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def read(self, timeout: float) -> TItem:
        """
        Return the next item, or raise an exception.

        All exceptions are treated as **retryable** by default — the observable
        will count consecutive failures and only terminate the stream once
        *max_consecutive_errors* is exceeded.

        Raise :class:`~py3r.media.types.FatalReadError` to signal an
        unrecoverable failure that should terminate the stream immediately,
        bypassing the retry counter entirely.
        """
        ...


def reader_observable(
    reader: IReader[TItem],
    read_timeout_seconds: float = 0.5,
    max_consecutive_errors: Optional[int] = 10,
) -> Observable[TItem]:
    """
    Turn any blocking reader object into an Observable sequence.  At
    subscription time a background thread is started to read from the reader
    one item at a time.

    :param reader: The reader to wrap.  Must implement ``open()``,
        ``close()``, and ``read(timeout)``.
    :param read_timeout_seconds: Per-read timeout forwarded to
        ``reader.read()``.  Keeping this short (e.g. 0.5 s) allows the loop
        to react quickly to disposal.
    :param max_consecutive_errors: How many errors in a row are tolerated
        before the stream is terminated with ``on_error``.  Pass ``None`` to
        retry forever (use with caution).  Defaults to **10** — roughly 5 s of
        continuous failure at the default timeout.  A successful read resets
        the counter to zero.  :class:`~py3r.media.types.FatalReadError` always
        terminates the stream immediately, regardless of this value.
    :return: Cold observable of items produced by *reader*.
    """
    def _subscribe(observer: ObserverBase[TItem], _=None) -> Disposable:
        stop = threading.Event()
        done = threading.Event()

        def run():
            consecutive_errors = 0

            try:
                reader.open()

                while not stop.is_set():
                    try:
                        item = reader.read(read_timeout_seconds)
                        consecutive_errors = 0  # reset on success
                    except FatalReadError as ex:
                        if not stop.is_set():
                            observer.on_error(ex)
                        return
                    except Exception as ex:
                        if stop.is_set():
                            return
                        consecutive_errors += 1
                        log.warning(
                            "Retryable read error (%d/%s): %s",
                            consecutive_errors,
                            str(max_consecutive_errors) if max_consecutive_errors is not None else "∞",
                            ex,
                        )
                        if max_consecutive_errors is not None and consecutive_errors >= max_consecutive_errors:
                            observer.on_error(
                                RuntimeError(
                                    f"Giving up after {consecutive_errors} consecutive "
                                    f"errors; last error: {ex}"
                                )
                            )
                            return
                        continue

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






