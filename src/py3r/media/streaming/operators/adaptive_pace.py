import collections
import queue
import threading
import time
from typing import Callable, Deque, Optional, TypeVar

import reactivex as rx
from reactivex.disposable import Disposable

_T = TypeVar("_T")


def adaptive_pace(
    window_size: int = 64,
    initial_interval: Optional[float] = None,
    *,
    min_interval: float = 1e-4,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.Observable[_T]]:
    """
    Smooth a bursty observable by estimating the average input interval from
    the last `window_size` inter-arrival intervals, then draining a FIFO queue
    at that pace.

    Guarantees:
      - no dropped items before disposal
      - FIFO order, assuming upstream serializes observer calls
      - error/completed are delivered after already-buffered items

    Notes:
      - Emits from an internal worker thread.
      - Use observe_on(...) after this operator if downstream needs a specific
        scheduler/thread.
      - Use subscribe_on(...) before this operator if upstream subscription
        needs a specific scheduler.
    """

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if initial_interval is not None and initial_interval < 0:
        raise ValueError("initial_interval must be non-negative or None")
    if min_interval <= 0:
        raise ValueError("min_interval must be positive")

    def _op(source: rx.abc.ObservableBase[_T]) -> rx.Observable[_T]:
        def _subscribe(
            observer: rx.abc.ObserverBase[_T],
            scheduler_: Optional[rx.abc.SchedulerBase] = None,
        ) -> Disposable:
            q: queue.Queue[tuple[str, object]] = queue.Queue()
            disposed = threading.Event()

            state_lock = threading.Lock()
            last_arrival: Optional[float] = None

            intervals: Deque[float] = collections.deque(maxlen=window_size)
            interval_sum = 0.0

            if initial_interval is not None:
                seed = max(initial_interval, min_interval)
                intervals.extend([seed] * window_size)
                interval_sum = seed * window_size

            def append_interval(dt: float) -> None:
                nonlocal interval_sum

                dt = max(dt, min_interval)

                with state_lock:
                    if len(intervals) == intervals.maxlen:
                        interval_sum -= intervals[0]
                    intervals.append(dt)
                    interval_sum += dt

            def current_period() -> float:
                with state_lock:
                    if intervals:
                        return max(interval_sum / len(intervals), min_interval)
                    return min_interval

            def on_next(x: _T) -> None:
                nonlocal last_arrival

                if disposed.is_set():
                    return

                now = time.perf_counter()

                with state_lock:
                    previous = last_arrival
                    last_arrival = now

                if previous is not None:
                    append_interval(now - previous)

                q.put(("next", x))

            def on_error(err: Exception) -> None:
                q.put(("error", err))

            def on_completed() -> None:
                q.put(("completed", None))

            def interruptible_sleep(seconds: float) -> bool:
                deadline = time.perf_counter() + seconds
                while not disposed.is_set():
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return True
                    disposed.wait(min(remaining, 0.05))
                return False

            def emit_loop() -> None:
                next_due = time.perf_counter()

                try:
                    while not disposed.is_set():
                        try:
                            kind, value = q.get(timeout=0.05)
                        except queue.Empty:
                            continue

                        if kind == "stop":
                            return

                        if kind == "next":
                            now = time.perf_counter()
                            if next_due > now:
                                if not interruptible_sleep(next_due - now):
                                    return

                            if disposed.is_set():
                                return

                            observer.on_next(value)  # type: ignore[arg-type]

                            p = current_period()
                            now = time.perf_counter()

                            # Keep emission starts roughly `p` seconds apart.
                            # If downstream is slow, do not build infinite catch-up debt.
                            next_due = max(next_due + p, now)

                        elif kind == "error":
                            if not disposed.is_set():
                                observer.on_error(value)  # type: ignore[arg-type]
                            return

                        elif kind == "completed":
                            if not disposed.is_set():
                                observer.on_completed()
                            return

                except Exception as e:
                    if not disposed.is_set():
                        observer.on_error(e)

            worker = threading.Thread(
                target=emit_loop,
                name="adaptive_pace_emit_loop",
                daemon=True,
            )
            worker.start()

            try:
                src_disp = source.subscribe(
                    on_next,
                    on_error,
                    on_completed,
                    scheduler=scheduler_,
                )
            except Exception as e:
                q.put(("error", e))
                src_disp = Disposable()

            def dispose() -> None:
                disposed.set()
                src_disp.dispose()

                with q.mutex:
                    q.queue.clear()

                q.put(("stop", None))

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _op
