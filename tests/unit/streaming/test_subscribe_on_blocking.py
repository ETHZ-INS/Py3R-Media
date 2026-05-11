"""Unit tests for subscribe_on_blocking operator."""
from __future__ import annotations
import threading
import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import TimeoutScheduler
from py3r.media.streaming.operators.subscribe_on_blocking import subscribe_on_blocking
from tests.unit.streaming.conftest import collect, from_items
SCHED = TimeoutScheduler.singleton()
class TestBlocking:
    def test_subscribe_blocks_until_subscribed(self):
        """subscribe() must not return before the upstream subscription is established."""
        subscribed_event = threading.Event()
        def recording_source_sub(observer, _=None):
            subscribed_event.set()
            observer.on_completed()
            return Disposable()
        source = rx.create(recording_source_sub)
        # subscribe() should block while the scheduler action runs
        collect(source.pipe(subscribe_on_blocking(SCHED)))
        assert subscribed_event.is_set()
    def test_subscribe_returns_after_subscription_created(self):
        """After subscribe() returns, is_subscribed must be True."""
        is_subscribed = [False]
        def source_sub(observer, _=None):
            is_subscribed[0] = True
            observer.on_completed()
            return Disposable()
        source = rx.create(source_sub)
        items, errors, _ = collect(source.pipe(subscribe_on_blocking(SCHED)))
        assert is_subscribed[0]
    def test_subscription_happens_on_scheduler_thread(self):
        main_tid = threading.get_ident()
        subscription_tid = [None]
        def source_sub(observer, _=None):
            subscription_tid[0] = threading.get_ident()
            observer.on_completed()
            return Disposable()
        source = rx.create(source_sub)
        collect(source.pipe(subscribe_on_blocking(SCHED)))
        assert subscription_tid[0] is not None
        assert subscription_tid[0] != main_tid
class TestDelivery:
    def test_items_flow_after_subscription(self):
        items, errors, _ = collect(from_items(1, 2, 3).pipe(subscribe_on_blocking(SCHED)))
        assert items == [1, 2, 3] and errors == []
    def test_error_flows_through(self):
        err = RuntimeError("oops")
        _, errors, _ = collect(from_items(error=err).pipe(subscribe_on_blocking(SCHED)))
        assert errors[0] is err
class TestDispose:
    def test_dispose_stops_subscription(self):
        disp = rx.never().pipe(subscribe_on_blocking(SCHED)).subscribe()
        disp.dispose()  # must not hang
