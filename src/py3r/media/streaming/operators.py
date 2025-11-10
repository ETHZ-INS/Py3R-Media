from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable, TypeVar, Optional
from threading import Lock

import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import Disposable, CompositeDisposable, SerialDisposable
from reactivex.scheduler import EventLoopScheduler

_T = TypeVar("_T")

def notify_future(
    future: Future[None],
    *,
    cancel_on_dispose: bool = True,
) -> Callable[[rx.Observable[_T]], rx.Observable[_T]]:
    """
    Operator that wires a Future to the subscription lifecycle.

    - On source `on_completed()`: future.set_result(None)
    - On source `on_error(e)`:    future.set_exception(e)
    - If downstream `observer.on_next(x)` raises: set_exception(e) and propagate on_error(e)
    - On `dispose()`:             future.cancel() (if `cancel_on_dispose=True`)

    Args:
        future: a concurrent.futures.Future[None] you created.
        cancel_on_dispose: if True, cancel the future on subscription dispose.

    Returns:
        A function mapping Observable[T] -> Observable[T].
    """

    lock = Lock()

    def _set_ok() -> None:
        with lock:
            if not future.done():
                future.set_result(None)

    def _set_err(e: Exception) -> None:
        with lock:
            if not future.done():
                future.set_exception(e)

    def _cancel() -> None:
        with lock:
            if cancel_on_dispose and not future.done():
                future.cancel()

    def _op(source: rx.Observable[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], scheduler: Optional[rx.abc.SchedulerBase] = None) -> rx.abc.DisposableBase:
            # Wrap downstream so we can catch on_next exceptions
            def _on_next(x: _T) -> None:
                try:
                    observer.on_next(x)
                except Exception as e:
                    # Downstream misbehaved; translate to error
                    observer.on_error(e)
                    _set_err(e)

            def _on_error(err: Exception) -> None:
                try:
                    observer.on_error(err)
                finally:
                    _set_err(err)

            def _on_completed() -> None:
                try:
                    observer.on_completed()
                finally:
                    _set_ok()

            upstream = source.subscribe(
                on_next=_on_next,
                on_error=_on_error,
                on_completed=_on_completed,
                scheduler=scheduler,
            )

            # Wrap the upstream disposable to intercept dispose()
            class _DisposeWrapper(Disposable):
                def __init__(self, inner: rx.abc.DisposableBase) -> None:
                    super().__init__()
                    self._inner = inner
                    self._disposed = False

                def dispose(self) -> None:
                    if not self._disposed:
                        self._disposed = True
                        try:
                            self._inner.dispose()
                        finally:
                            _cancel()

            return _DisposeWrapper(upstream)

        return rx.create(_subscribe)

    return _op


def observe_on_bounded(scheduler, maxsize=256, policy="block") -> Callable[[rx.Observable[_T]], rx.Observable[_T]]:
    """
    Like observe_on, but with a bounded internal queue.
    policy: "block" | "drop_newest" | "drop_oldest"
    """
    assert policy in ("block", "drop_newest", "drop_oldest")

    def _op(source: rx.Observable[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], _scheduler: Optional[rx.abc.SchedulerBase] = None) -> rx.abc.DisposableBase:
            q = queue.Queue(maxsize=maxsize)
            stop = threading.Event()
            wdisp = SerialDisposable()

            def enqueue_action(action):
                #if q.full():
                #    print(f"{name} queue full: size={q.qsize()} policy={policy}")
                if policy == "block":
                    # May block when full: real backpressure
                    q.put(action)
                elif policy == "drop_newest":
                    try:
                        q.put_nowait(action)
                    except queue.Full:
                        # drop newest
                        pass
                else:  # drop_oldest
                    try:
                        q.put_nowait(action)
                    except queue.Full:
                        try:
                            _ = q.get_nowait()  # evict one
                        except queue.Empty:
                            pass
                        try:
                            q.put_nowait(action)
                        except queue.Full:
                            # if a race refilled it, just drop
                            pass

            # Worker that runs on the target scheduler
            def worker(_sc=None, _st=None):
                if stop.is_set():
                    return
                try:
                    try:
                        action = q.get(timeout=0.05)
                    except queue.Empty:
                        # reschedule to check again
                        wdisp.disposable = scheduler.schedule(worker)
                        return
                    try:
                        action()
                    finally:
                        # reschedule immediately to drain
                        wdisp.disposable = scheduler.schedule(worker)
                except Exception as e:
                    # Forward unexpected worker exceptions
                    observer.on_error(e)

            # Upstream subscription: enqueue observer actions
            def on_next(x):
                def act():
                    observer.on_next(x)
                enqueue_action(act)

            def on_error(e):
                def act():
                    observer.on_error(e)
                enqueue_action(act)

            def on_completed():
                def act():
                    observer.on_completed()
                enqueue_action(act)

            upstream = source.subscribe(on_next, on_error, on_completed, scheduler=_scheduler)
            # start the worker on target scheduler
            wdisp.disposable = scheduler.schedule(worker)

            def dispose():
                stop.set()
                wdisp.dispose()
                upstream.dispose()
                # unblock worker if waiting
                try: q.put_nowait(lambda: None)
                except Exception: pass

            return CompositeDisposable(upstream, wdisp, Disposable(dispose))
        return rx.create(_subscribe)
    return _op


def adaptive_pace(
    initial_interval: Optional[float] = None,
    learn_rate: float = 0.1,
    scheduler: Optional[rx.abc.SchedulerBase] = None,
) -> Callable[[rx.Observable[_T]], rx.Observable[_T]]:
    """
    Normalize pacing of an observable sequence to a steady rate inferred from input timing.

    Args:
        initial_interval: Initial guess for frame period in seconds (e.g. 1/30 for 30fps).
        learn_rate: Exponential moving average factor for adapting the interval.
        scheduler: Optional scheduler (defaults to a new thread).

    Returns:
        Operator: Observable[T] -> Observable[T]
    """
    if scheduler is None:
        scheduler = EventLoopScheduler()

    def _op(source: rx.Observable[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], _scheduler: Optional[rx.abc.SchedulerBase] = None) -> rx.abc.DisposableBase:
            q: queue.Queue[_T] = queue.Queue()
            last_time: Optional[float] = None
            period: float = initial_interval or 0.0
            stopped = threading.Event()
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
                q.put(x)

            def on_error(err: Exception):
                stopped.set()
                observer.on_error(err)

            def on_completed():
                stopped.set()

            src_disp = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)

            def emit_loop():
                try:
                    """Emit items at the learned interval until source completes or disposed."""
                    nonlocal period
                    while not disposed.is_set():
                        # Try to get a frame
                        try:
                            item = q.get(timeout=0.01)
                            observer.on_next(item)
                        except queue.Empty:
                            pass

                        if stopped.is_set() and q.empty():
                            observer.on_completed()
                            return

                        # Sleep for the current period (split into small chunks)
                        sleep_time = max(period, 1e-6) if period > 0 else 0.001
                        deadline = time.perf_counter() + sleep_time
                        while True:
                            if disposed.is_set():
                                return
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                break
                            time.sleep(min(0.002, remaining))
                except Exception as e:
                    import traceback
                    traceback.print_exc()

            # Schedule the emitter on its own thread
            emit_disp = scheduler.schedule(lambda *_: emit_loop())

            def dispose():
                disposed.set()
                src_disp.dispose()
                with q.mutex:
                    q.queue.clear()

            return CompositeDisposable(src_disp, emit_disp, Disposable(dispose))

        return rx.create(_subscribe)
    return _op
