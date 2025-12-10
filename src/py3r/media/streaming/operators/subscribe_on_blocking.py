import threading
from typing import Callable, Optional, Any, TypeVar

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import SingleAssignmentDisposable, SerialDisposable, ScheduledDisposable


_T = TypeVar("_T")


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
