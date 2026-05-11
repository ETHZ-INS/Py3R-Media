"""Unit tests for subscribe_on_future operator."""
from __future__ import annotations
from concurrent.futures import Future
import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import TimeoutScheduler
from py3r.media.streaming.operators.subscribe_on_future import subscribe_on_future
from tests.unit.streaming.conftest import collect, from_items
SCHED = TimeoutScheduler.singleton()
class TestSubscribed:
    def test_subscribed_future_resolves(self):
        subscribed: Future[None] = Future()
        collect(from_items(1).pipe(subscribe_on_future(SCHED, subscribed=subscribed)))
        assert subscribed.done() and subscribed.result() is None
    def test_items_flow_after_subscribed_resolves(self):
        subscribed: Future[None] = Future()
        items, errors, _ = collect(from_items(1, 2, 3).pipe(subscribe_on_future(SCHED, subscribed=subscribed)))
        assert subscribed.done()
        assert items == [1, 2, 3]
    def test_subscribed_resolves_after_subscription_made(self):
        """Core invariant: future must not be set until source.subscribe() has returned.

        Callers block on subscribed.result() before producing data, so the
        future must resolve AFTER the upstream subscription is fully established.
        """
        subscribed: Future[None] = Future()
        was_done_during_subscribe: list[bool] = []

        def recording_source_sub(observer, _=None):
            # Capture the future state while subscribe_on_future is still
            # inside source.subscribe(observer) — future must not be set yet.
            was_done_during_subscribe.append(subscribed.done())
            observer.on_completed()
            return Disposable()

        source = rx.create(recording_source_sub)
        collect(source.pipe(subscribe_on_future(SCHED, subscribed=subscribed)))

        assert was_done_during_subscribe == [False], (
            "subscribed future resolved too early — callers relying on it as a "
            "post-subscription signal will open their source before the pipeline is ready"
        )
        assert subscribed.done() and subscribed.result() is None
class TestDisposed:
    def test_disposed_future_resolves_after_dispose(self):
        subscribed: Future[None] = Future()
        disposed: Future[None] = Future()
        disp = rx.never().pipe(
            subscribe_on_future(SCHED, subscribed=subscribed, disposed=disposed)
        ).subscribe()
        subscribed.result(timeout=2.0)  # wait for subscription
        assert not disposed.done()
        disp.dispose()
        disposed.result(timeout=2.0)
        assert disposed.done()
    def test_disposed_future_not_resolved_before_dispose(self):
        subscribed: Future[None] = Future()
        disposed: Future[None] = Future()
        disp = rx.never().pipe(
            subscribe_on_future(SCHED, subscribed=subscribed, disposed=disposed)
        ).subscribe()
        subscribed.result(timeout=2.0)
        assert not disposed.done()
        disp.dispose()
class TestFreshFutures:
    def test_each_call_creates_independent_futures(self):
        """Calling subscribe_on_future() twice must not share state."""
        op = subscribe_on_future(SCHED)
        subscribed1: Future = Future()
        subscribed2: Future = Future()
        op1 = subscribe_on_future(SCHED, subscribed=subscribed1)
        op2 = subscribe_on_future(SCHED, subscribed=subscribed2)
        collect(from_items(1).pipe(op1))
        assert subscribed1.done()
        assert not subscribed2.done()  # op2 was never subscribed
class TestSubscribeError:
    def test_subscribe_error_propagates_to_subscribed_future(self):
        subscribed: Future[None] = Future()
        err = RuntimeError("subscribe boom")
        def bad_source_sub(observer, _=None):
            raise err
        source = rx.create(bad_source_sub)
        try:
            collect(source.pipe(subscribe_on_future(SCHED, subscribed=subscribed)))
        except Exception:
            pass
        subscribed.exception(timeout=2.0)
        assert subscribed.done()
