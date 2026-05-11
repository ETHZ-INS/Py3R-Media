"""
Shared helpers for streaming operator unit tests.
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional, Tuple

import pytest
import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import TimeoutScheduler


# ---------------------------------------------------------------------------
# Scheduler fixture  (module-scoped — TimeoutScheduler is a singleton anyway)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scheduler():
    return TimeoutScheduler.singleton()


# ---------------------------------------------------------------------------
# Helpers (importable by test modules)
# ---------------------------------------------------------------------------

def collect(
    source: rx.Observable,
    *,
    timeout: float = 2.0,
) -> Tuple[List[Any], List[BaseException], rx.abc.DisposableBase]:
    """
    Subscribe to *source*, collect every on_next value and any on_error into
    lists, and block until on_completed / on_error fires or *timeout* expires.

    Returns ``(items, errors, disposable)``.
    """
    items: List[Any] = []
    errors: List[BaseException] = []
    done = threading.Event()

    disp = source.subscribe(
        on_next=items.append,
        on_error=lambda e: (errors.append(e), done.set()),
        on_completed=done.set,
    )
    done.wait(timeout=timeout)
    return items, errors, disp


def from_items(*items: Any, error: Optional[BaseException] = None) -> rx.Observable:
    """
    Cold observable that emits *items* synchronously on subscribe, then
    either calls on_completed or on_error(*error*).
    """
    def subscribe(observer, _=None):
        for x in items:
            observer.on_next(x)
        if error is not None:
            observer.on_error(error)
        else:
            observer.on_completed()
        return Disposable()

    return rx.create(subscribe)

