"""Unit tests for adaptive_pace operator."""
from __future__ import annotations
import time
import threading
from reactivex.scheduler import TimeoutScheduler
from py3r.media.streaming.operators.adaptive_pace import adaptive_pace
from tests.unit.streaming.conftest import collect, from_items
SCHED = TimeoutScheduler.singleton()
POLL = 0.001  # fast poll for tests
class TestDelivery:
    def test_all_items_delivered(self):
        items, errors, _ = collect(
            from_items(1, 2, 3, 4, 5).pipe(adaptive_pace(_poll=POLL))
        )
        assert items == [1, 2, 3, 4, 5] and errors == []
    def test_items_delivered_in_order(self):
        N = 10
        items, _, _ = collect(from_items(*range(N)).pipe(adaptive_pace(_poll=POLL)))
        assert items == list(range(N))
    def test_empty_source_completes(self):
        done = threading.Event()
        from_items().pipe(adaptive_pace(_poll=POLL)).subscribe(
            on_next=lambda x: None, on_completed=done.set
        )
        assert done.wait(timeout=2.0)
class TestError:
    def test_error_propagates(self):
        err = RuntimeError("source error")
        _, errors, _ = collect(from_items(1, 2, error=err).pipe(adaptive_pace(_poll=POLL)))
        assert errors[0] is err
    def test_items_before_error_are_delivered(self):
        """Items already buffered before the error must be delivered first."""
        err = RuntimeError("late error")
        items, errors, _ = collect(from_items(1, 2, error=err).pipe(adaptive_pace(_poll=POLL)))
        assert 1 in items and 2 in items
        assert len(errors) == 1
class TestPacing:
    def test_initial_interval_delays_delivery(self):
        """With a 50ms initial interval, 3 items should take at least ~100ms."""
        t0 = time.perf_counter()
        items, _, _ = collect(
            from_items(1, 2, 3).pipe(adaptive_pace(initial_interval=0.05, _poll=POLL))
        )
        elapsed = time.perf_counter() - t0
        assert items == [1, 2, 3]
        assert elapsed >= 0.08  # generous lower bound
class TestDispose:
    def test_dispose_stops_delivery(self):
        delivered = []
        gate = threading.Event()
        import reactivex as rx
        from reactivex.subject import Subject
        subj = Subject()
        disp = subj.pipe(adaptive_pace(_poll=POLL)).subscribe(on_next=delivered.append)
        subj.on_next(1)
        time.sleep(0.02)
        disp.dispose()
        subj.on_next(2)
        time.sleep(0.05)
        assert 2 not in delivered
