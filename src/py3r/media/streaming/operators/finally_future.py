from concurrent.futures import Future
from typing import Callable, Optional, TypeVar

import reactivex as rx
from reactivex.disposable import Disposable


_T = TypeVar("_T")


def finally_future(
    future: Future[None],
    *,
    cancel_on_dispose: bool = False,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.abc.ObservableBase[_T]]:
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
            future.set_result(None)

    def _set_err(e: Exception) -> None:
        if not future.done():
            future.set_exception(e)

    def _cancel() -> None:
        if not future.done():
            future.cancel()

    def _op(source: rx.abc.ObservableBase[_T]) -> rx.Observable[_T]:
        def _subscribe(observer: rx.abc.ObserverBase[_T], scheduler: Optional[rx.abc.SchedulerBase] = None) -> Disposable:
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
                try:
                    nonlocal error
                    error = err
                    observer.on_error(err)
                finally:
                    _set_err(err)

            def _on_completed() -> None:
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
