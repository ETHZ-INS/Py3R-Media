import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable, TypeVar, Optional, Any

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable, CompositeDisposable, SerialDisposable, ScheduledDisposable, \
    SingleAssignmentDisposable
from reactivex.scheduler import EventLoopScheduler

_T = TypeVar("_T")

def finally_future(
    future: Future[None],
    *,
    cancel_on_dispose: bool = False,
) -> Callable[[rx.Observable[_T]], rx.Observable[_T]]:
    """
    Operator that wires a Future to the subscription lifecycle.

    - On source `on_completed()`: future.set_result(None)
    - On source `on_error(e)`:    future.set_exception(e)
    - If downstream `observer.on_next(x)` raises: set_exception(e) and propagate on_error(e)
    - On `dispose()`:             future.cancel() (if `cancel_on_dispose=True`)

    Args:
        future: a concurrent.futures.Future[None] you created.
        cancel_on_dispose: if True, cancel the future if the subscription is disposed before a terminal signal.

    Returns:
        A function mapping Observable[T] -> Observable[T].
    """

    def _set_ok() -> None:
        if not future.done():
            print("Setting future result to None")
            future.set_result(None)

    def _set_err(e: Exception) -> None:
        if not future.done():
            print("Setting future result to exception:", e)
            future.set_exception(e)

    def _cancel() -> None:
        if not future.done():
            print("Cancelling future")
            future.cancel()

    def _op(source: rx.Observable[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], scheduler: Optional[rx.abc.SchedulerBase] = None) -> rx.abc.DisposableBase:
            completed = False
            error: Optional[Exception] = None

            # Wrap downstream so we can catch on_next exceptions
            def _on_next(x: _T) -> None:
                try:
                    observer.on_next(x)
                except Exception as e:
                    # Downstream misbehaved; translate to error
                    observer.on_error(e)
                    _set_err(e)

            def _on_error(err: Exception) -> None:
                print("_notify_future: upstream on_error:", err, "on thread", threading.current_thread().name)
                try:
                    nonlocal error
                    error = err
                    observer.on_error(err)
                finally:
                    _set_err(err)

            def _on_completed() -> None:
                print("_notify_future: upstream on_completed on thread", threading.current_thread().name)
                try:
                    nonlocal completed
                    completed = True
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
                    print("_notify_future: dispose on thread", threading.current_thread().name)
                    if not self._disposed:
                        self._disposed = True
                        try:
                            self._inner.dispose()
                        finally:
                            if error:
                                _set_err(error)
                            elif completed:
                                _set_ok()
                            elif cancel_on_dispose:
                                _cancel()
                            else:
                                _set_ok()

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


def subscribe_on_blocking(
    scheduler: rx.abc.SchedulerBase,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.abc.ObservableBase[_T]]:
    def _subscribe_on_blocking(source: rx.abc.ObservableBase[_T]) -> rx.abc.ObservableBase[_T]:
        """
        Like subscribe_on, but the outer subscribe() blocks until the
        subscription has actually been created on the target scheduler.

        - Subscription side effects (resource creation) run on `scheduler`
        - Unsubscription (dispose) is also scheduled on `scheduler`
          via ScheduledDisposable
        - The caller's subscribe(...) does not return until the scheduled
          subscription action has run.
        """

        def subscribe(
            observer: rx.abc.ObserverBase[_T],
            _: Optional[rx.abc.SchedulerBase] = None,
        ):
            m = SingleAssignmentDisposable()
            d = SerialDisposable()
            d.disposable = m

            # Gate to signal that the scheduled subscription has run
            gate = threading.Event()

            def action(
                sched: rx.abc.SchedulerBase,
                _state: Optional[Any] = None,
            ):
                try:
                    # Create the real subscription on the scheduler thread
                    inner_disp = source.subscribe(observer)
                    # Ensure its disposal also happens on `sched`
                    d.disposable = ScheduledDisposable(sched, inner_disp)
                finally:
                    # Always release the gate, even if subscribe() throws
                    gate.set()

            # Schedule the subscription on the target scheduler
            m.disposable = scheduler.schedule(action)

            # Block caller until the above action has executed
            gate.wait()

            # From the caller's perspective, subscription is now fully established
            return d

        return Observable(subscribe)

    return _subscribe_on_blocking


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
