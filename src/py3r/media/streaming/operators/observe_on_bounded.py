import queue
import threading
from typing import Callable, Optional, TypeVar

import reactivex as rx
from reactivex.disposable import SerialDisposable, CompositeDisposable, Disposable


_T = TypeVar("_T")


def observe_on_bounded(
    scheduler,
    maxsize: int = 256,
    policy: str = "block",
    timeout: float = 0.1,
    _worker_poll: float = 0.05,
) -> Callable[[rx.Observable[_T]], rx.Observable[_T]]:
    """
    Like observe_on, but with a bounded internal queue.

    policy:
      - "block": block producer until queue has room, but wake on dispose
      - "drop_newest": drop the incoming item if full
      - "drop_oldest": evict one queued item if full, then enqueue newest
    """
    if policy not in ("block", "drop_newest", "drop_oldest"):
        raise ValueError(f"invalid policy: {policy}")

    def _op(source: rx.Observable[_T]) -> rx.Observable[_T]:
        def _subscribe(observer, _scheduler=None):
            q: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=maxsize)
            disposed = threading.Event()
            worker_disp = SerialDisposable()
            upstream_disp = SerialDisposable()

            # Prevent multiple terminal deliveries.
            terminal_sent = threading.Event()

            def enqueue_action(action: Callable[[], None], *, terminal: bool = False) -> bool:
                if disposed.is_set():
                    return False

                if policy == "block":
                    # IMPORTANT: never block forever.
                    while not disposed.is_set():
                        try:
                            q.put(action, timeout=timeout)
                            return True
                        except queue.Full:
                            continue
                    return False

                if policy == "drop_newest":
                    try:
                        q.put_nowait(action)
                        return True
                    except queue.Full:
                        # For terminal signals you may prefer to force them through.
                        if terminal and not disposed.is_set():
                            try:
                                _ = q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                q.put_nowait(action)
                                return True
                            except queue.Full:
                                pass
                        return False

                # drop_oldest
                try:
                    q.put_nowait(action)
                    return True
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(action)
                        return True
                    except queue.Full:
                        return False

            def worker(_sc=None, _st=None):
                if disposed.is_set():
                    return

                try:
                    try:
                        action = q.get(timeout=_worker_poll)
                    except queue.Empty:
                        if not disposed.is_set():
                            worker_disp.disposable = scheduler.schedule(worker)
                        return

                    if disposed.is_set():
                        return

                    action()

                except Exception as e:
                    if not disposed.is_set():
                        observer.on_error(e)
                    return

                if not disposed.is_set():
                    worker_disp.disposable = scheduler.schedule(worker)

            def on_next(x: _T):
                if disposed.is_set() or terminal_sent.is_set():
                    return

                def act():
                    if not disposed.is_set():
                        observer.on_next(x)

                enqueue_action(act, terminal=False)

            def on_error(err: Exception):
                if disposed.is_set() or terminal_sent.is_set():
                    return
                terminal_sent.set()

                def act():
                    if not disposed.is_set():
                        observer.on_error(err)

                enqueue_action(act, terminal=True)

            def on_completed():
                if disposed.is_set() or terminal_sent.is_set():
                    return
                terminal_sent.set()

                def act():
                    if not disposed.is_set():
                        observer.on_completed()

                enqueue_action(act, terminal=True)

            upstream_disp.disposable = source.subscribe(
                on_next,
                on_error,
                on_completed,
                scheduler=_scheduler,
            )

            worker_disp.disposable = scheduler.schedule(worker)

            def dispose():
                if disposed.is_set():
                    return

                disposed.set()

                # Stop upstream first so no new items arrive.
                upstream_disp.dispose()

                # Wake up any blocked enqueue_action("block") loop by making room
                # and/or nudging the worker.
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass

                try:
                    q.put_nowait(lambda: None)
                except queue.Full:
                    pass

                worker_disp.dispose()

            return CompositeDisposable(
                upstream_disp,
                worker_disp,
                Disposable(dispose),
            )

        return rx.create(_subscribe)

    return _op
