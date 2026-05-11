"""Unit tests for observe_on_bounded operator."""
from __future__ import annotations
import threading
import time
import reactivex as rx
from reactivex.subject import Subject
from reactivex.scheduler import TimeoutScheduler
import pytest
from py3r.media.streaming.operators.observe_on_bounded import observe_on_bounded
from tests.unit.streaming.conftest import collect, from_items
SCHED = TimeoutScheduler.singleton()
POLL = 0.002  # fast poll for tests
class TestBasic:
    def test_all_items_delivered(self):
        items, errors, _ = collect(
            from_items(1, 2, 3).pipe(observe_on_bounded(SCHED, _worker_poll=POLL))
        )
        assert items == [1, 2, 3] and errors == []
    def test_completed_delivered(self):
        done = threading.Event()
        from_items(1).pipe(observe_on_bounded(SCHED, _worker_poll=POLL)).subscribe(
            on_next=lambda x: None, on_completed=done.set
        )
        assert done.wait(timeout=2.0)
    def test_error_delivered(self):
        err = RuntimeError("boom")
        _, errors, _ = collect(from_items(error=err).pipe(observe_on_bounded(SCHED, _worker_poll=POLL)))
        assert errors[0] is err
    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="invalid policy"):
            observe_on_bounded(SCHED, policy="unknown")
class TestDeliveryThread:
    def test_items_delivered_on_different_thread(self):
        main_tid = threading.get_ident()
        delivery_tids = []
        done = threading.Event()
        from_items(1, 2, 3).pipe(
            observe_on_bounded(SCHED, _worker_poll=POLL)
        ).subscribe(
            on_next=lambda x: delivery_tids.append(threading.get_ident()),
            on_completed=done.set,
        )
        done.wait(timeout=2.0)
        assert all(tid != main_tid for tid in delivery_tids)
class TestBlockPolicy:
    def test_block_delivers_all_items(self):
        """Emit from a background thread so the worker can interleave with the producer."""
        N = 20
        received = []
        done = threading.Event()
        subj = Subject()
        subj.pipe(
            observe_on_bounded(SCHED, maxsize=4, policy="block", timeout=0.1, _worker_poll=POLL)
        ).subscribe(on_next=received.append, on_completed=done.set)

        def emit():
            for i in range(N):
                subj.on_next(i)
            subj.on_completed()

        t = threading.Thread(target=emit, daemon=True)
        t.start()
        done.wait(timeout=5.0)
        t.join(timeout=1.0)
        assert received == list(range(N))
class TestDropNewest:
    def test_drop_newest_drops_when_full(self):
        """Block worker on first item, then overflow the queue � extra items must be dropped."""
        MAXSIZE = 2
        received = []
        first_arrived = threading.Event()
        release_first = threading.Event()
        done = threading.Event()
        subj = Subject()
        def slow_on_next(x):
            if not first_arrived.is_set():
                first_arrived.set()
                release_first.wait(timeout=2.0)
            received.append(x)
        subj.pipe(
            observe_on_bounded(SCHED, maxsize=MAXSIZE, policy="drop_newest", _worker_poll=POLL)
        ).subscribe(on_next=slow_on_next, on_completed=done.set)
        # Emit item 0 � worker picks it up and blocks
        subj.on_next(0)
        first_arrived.wait(timeout=2.0)
        # Queue is now empty (worker removed item 0). Emit MAXSIZE+3 more items.
        for i in range(1, MAXSIZE + 4):
            subj.on_next(i)
        # Items 1..MAXSIZE fill the queue; items beyond are dropped.
        release_first.set()
        subj.on_completed()
        done.wait(timeout=2.0)
        assert 0 in received
        assert len(received) <= MAXSIZE + 1  # 0 + at most MAXSIZE queued
class TestDropOldest:
    def test_drop_oldest_newest_survives(self):
        """With drop_oldest, the newest item always overwrites the oldest queued item."""
        MAXSIZE = 1
        received = []
        first_arrived = threading.Event()
        release_first = threading.Event()
        item3_received = threading.Event()
        done = threading.Event()
        subj = Subject()
        def slow_on_next(x):
            if not first_arrived.is_set():
                first_arrived.set()
                release_first.wait(timeout=2.0)
            received.append(x)
            if x == 3:
                item3_received.set()
        subj.pipe(
            observe_on_bounded(SCHED, maxsize=MAXSIZE, policy="drop_oldest", _worker_poll=POLL)
        ).subscribe(on_next=slow_on_next, on_completed=done.set)
        subj.on_next(0)
        first_arrived.wait(timeout=2.0)
        # Fill queue with 1, then overwrite with 2, then 3
        subj.on_next(1)
        subj.on_next(2)
        subj.on_next(3)
        release_first.set()
        # Wait for item 3 to be delivered before completing, to avoid a race
        # where on_completed evicts item 3 from the queue.
        item3_received.wait(timeout=2.0)
        subj.on_completed()
        done.wait(timeout=2.0)
        assert 0 in received
        # 3 must be present (it's the last written, so it replaced all earlier ones)
        assert 3 in received
class TestDispose:
    def test_dispose_stops_delivery(self):
        delivered = []
        slow_gate = threading.Event()
        subj = Subject()
        disp = subj.pipe(
            observe_on_bounded(SCHED, _worker_poll=POLL)
        ).subscribe(on_next=delivered.append)
        subj.on_next(1)
        time.sleep(0.05)
        disp.dispose()
        subj.on_next(2)
        time.sleep(0.05)
        assert 2 not in delivered
