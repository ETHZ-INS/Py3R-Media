import threading
from concurrent.futures import Future

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable, SingleAssignmentDisposable, SerialDisposable
from typing import Any, Optional, TypeVar, Callable


class FutureScheduledDisposable(Disposable):
    def __init__(self, scheduler, disp, disposed: Future[None]):
        super().__init__()
        self._scheduler = scheduler
        self._disp = disp
        self._disposed = disposed
        self._is_disposed = False
        self._lock = threading.Lock()

    def dispose(self) -> None:
        with self._lock:
            if self._is_disposed:
                return
            self._is_disposed = True

            def action(_sched, _state: Any = None):
                try:
                    self._disp.dispose()
                except BaseException as exc:
                    # disposal failed → surface as exception on the Future
                    if not self._disposed.done():
                        self._disposed.set_exception(exc)
                    raise
                else:
                    if not self._disposed.done():
                        self._disposed.set_result(None)

            self._scheduler.schedule(action)


_T = TypeVar("_T")

def subscribe_on_future(
    scheduler: rx.abc.SchedulerBase,
    subscribed: Optional[Future[None]] = None,
    disposed: Optional[Future[None]] = None,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.Observable[_T]]:
    # Create fresh Futures here (not in the signature) to avoid the
    # mutable-default-argument trap where all callers share the same object.
    if subscribed is None:
        subscribed = Future()
    if disposed is None:
        disposed = Future()

    def _op(source: rx.abc.ObservableBase[_T]) -> rx.Observable[_T]:

        def subscribe(
            observer: rx.abc.ObserverBase[_T],
            _: Optional[rx.abc.SchedulerBase] = None,
        ) -> rx.abc.DisposableBase:
            m = SingleAssignmentDisposable()
            d = SerialDisposable()
            d.disposable = m

            def action(
                    sched: rx.abc.SchedulerBase,
                    _state: Optional[Any] = None,
            ):
                try:
                    inner_disp = source.subscribe(observer)
                except BaseException as exc:
                    if not subscribed.done():
                        subscribed.set_exception(exc)
                    raise
                else:
                    # Subscription fully established — signal now so callers that
                    # blocked on subscribed.result() know it is safe to start producing.
                    d.disposable = FutureScheduledDisposable(sched, inner_disp, disposed)
                    if not subscribed.done():
                        subscribed.set_result(None)

            m.disposable = scheduler.schedule(action)
            return d

        return Observable(subscribe)

    return _op
