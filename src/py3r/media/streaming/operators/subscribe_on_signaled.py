import threading
from concurrent.futures import Future

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable, SingleAssignmentDisposable, SerialDisposable
from typing import Any, Optional, TypeVar, Callable


class SubscriptionSignal:
    """
    Exposes two Futures:

      - subscribed: completes when the subscription has been established
                    on the target scheduler.

      - disposed:   completes when the scheduled disposal has finished
                    on the target scheduler.

    You can use .result(timeout=...) or .add_done_callback(...) on both.
    """

    def __init__(self) -> None:
        self.subscribed: Future[None] = Future()
        self.disposed: Future[None] = Future()

    def _on_subscribed(self) -> None:
        if not self.subscribed.done():
            self.subscribed.set_result(None)

    def _on_subscribe_error(self, exc: BaseException) -> None:
        if not self.subscribed.done():
            self.subscribed.set_exception(exc)

    def _on_disposed(self) -> None:
        if not self.disposed.done():
            self.disposed.set_result(None)

    def _on_dispose_error(self, exc: BaseException) -> None:
        if not self.disposed.done():
            self.disposed.set_exception(exc)


class SignalScheduledDisposable(Disposable):
    def __init__(self, scheduler, disp, signal: SubscriptionSignal):
        super().__init__()
        self._scheduler = scheduler
        self._disp = disp
        self._signal = signal
        self._is_disposed = False
        self._lock = threading.Lock()

    def dispose(self) -> None:
        with self._lock:
            if self._is_disposed:
                return
            self._is_disposed = True

            def action(sched, _state: Any = None):
                try:
                    self._disp.dispose()
                except BaseException as exc:
                    # disposal failed → surface as exception on the Future
                    self._signal._on_dispose_error(exc)
                    raise
                else:
                    self._signal._on_disposed()

            self._scheduler.schedule(action)


_T = TypeVar("_T")

def subscribe_on_signaled(
    scheduler: rx.abc.SchedulerBase,
    signal: SubscriptionSignal,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.abc.ObservableBase[_T]]:

    def _op(source: rx.abc.ObservableBase[_T]) -> rx.abc.ObservableBase[_T]:

        def subscribe(
            observer: rx.abc.ObserverBase[_T],
            _: Optional[rx.abc.SchedulerBase] = None,
        ):
            m = SingleAssignmentDisposable()
            d = SerialDisposable()
            d.disposable = m

            def action(
                sched: rx.abc.SchedulerBase,
                _state: Optional[Any] = None,
            ):
                try:
                    # subscribe upstream on this scheduler
                    inner_disp = source.subscribe(observer)
                except BaseException as exc:
                    # subscription failed
                    signal._on_subscribe_error(exc)
                    raise
                else:
                    # subscription succeeded; disposal will be signaled separately
                    d.disposable = SignalScheduledDisposable(sched, inner_disp, signal)
                    signal._on_subscribed()

            m.disposable = scheduler.schedule(action)
            return d

        return Observable(subscribe)

    return _op
