import queue
import threading
from typing import Callable, Optional, TypeVar

import reactivex as rx
from reactivex.disposable import SerialDisposable, CompositeDisposable, Disposable


_T = TypeVar("_T")


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
                # noinspection PyBroadException
                try: q.put_nowait(lambda: None)
                except Exception: pass

            return CompositeDisposable(upstream, wdisp, Disposable(dispose))
        return rx.create(_subscribe)
    return _op
