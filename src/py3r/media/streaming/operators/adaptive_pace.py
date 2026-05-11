import queue
import threading
import time
from typing import TypeVar, Optional, Callable

import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import TimeoutScheduler

_T = TypeVar("_T")


def adaptive_pace(
    initial_interval: Optional[float] = None,
    learn_rate: float = 0.1,
    scheduler: Optional[rx.abc.SchedulerBase] = None,
    _poll: float = 0.01,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.Observable[_T]]:
    def _op(source: rx.abc.ObservableBase[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], scheduler_: Optional[rx.abc.SchedulerBase] = None) -> Disposable:
            _scheduler = scheduler or scheduler_ or TimeoutScheduler.singleton()

            q: queue.Queue = queue.Queue()
            last_time: Optional[float] = None
            period: float = initial_interval or 0.0
            disposed = threading.Event()

            def update_interval(now: float):
                nonlocal last_time, period
                if last_time is not None:
                    dt = now - last_time
                    if period <= 0:
                        period = dt
                    else:
                        period = (1 - learn_rate) * period + learn_rate * dt
                last_time = now

            def on_next(x: _T):
                update_interval(time.perf_counter())
                q.put(("next", x))

            def on_error(err: Exception):
                # Route through queue so buffered on_next items are delivered first.
                q.put(("error", err))

            def on_completed():
                q.put(("completed", None))

            src_disp = source.subscribe(
                on_next,
                on_error,
                on_completed,
                scheduler=_scheduler,
            )

            def emit_loop():
                try:
                    while not disposed.is_set():
                        try:
                            kind, value = q.get(timeout=_poll)
                        except queue.Empty:
                            continue

                        if kind == "next":
                            observer.on_next(value)
                        elif kind == "error":
                            observer.on_error(value)
                            return
                        else:  # completed
                            observer.on_completed()
                            return

                        # Pace: sleep for the learned interval, waking frequently
                        # so we can respond to dispose() quickly.
                        sleep_time = max(period, 1e-6)
                        deadline = time.perf_counter() + sleep_time
                        while not disposed.is_set():
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                break
                            time.sleep(min(remaining, _poll))
                except Exception as e:
                    observer.on_error(e)

            emit_disp = _scheduler.schedule(lambda sch, _: emit_loop())

            def dispose():
                disposed.set()
                src_disp.dispose()
                emit_disp.dispose()
                with q.mutex:
                    q.queue.clear()

            return Disposable(dispose)

        return rx.create(_subscribe)
    return _op
