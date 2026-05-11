"""Unit tests for reader_observable."""
from __future__ import annotations
import threading
from typing import List
import pytest
import reactivex as rx
from py3r.media.streaming.observables.reader_observable import reader_observable, IReader
from py3r.media.types import FatalReadError
from tests.unit.streaming.conftest import collect
class _FakeReader:
    """Emits items from a list, then raises a terminal exception."""
    def __init__(self, items, *, terminal=None, retryable_after: int = -1):
        self._items = list(items)
        self._terminal = terminal if terminal is not None else FatalReadError("eof")
        self._retryable_after = retryable_after  # raise RetryErr after this many reads (-1 = never)
        self._idx = 0
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0
    def open(self): self.open_count += 1
    def close(self): self.close_count += 1
    def read(self, timeout=0.5):
        self.read_count += 1
        if self._retryable_after >= 0 and self._idx >= self._retryable_after:
            raise RuntimeError("retryable error")
        if self._idx < len(self._items):
            v = self._items[self._idx]
            self._idx += 1
            return v
        raise self._terminal
class TestLifecycle:
    def test_open_called_on_subscribe(self):
        r = _FakeReader([1])
        collect(reader_observable(r))
        assert r.open_count == 1
    def test_close_called_after_completion(self):
        r = _FakeReader([1])
        collect(reader_observable(r))
        assert r.close_count == 1
    def test_close_called_even_on_fatal_error(self):
        r = _FakeReader([], terminal=FatalReadError("die"))
        collect(reader_observable(r))
        assert r.close_count == 1
    def test_close_called_on_dispose(self):
        stop = threading.Event()
        class BlockingReader:
            def open(self): pass
            def close(self): stop.set()
            def read(self, timeout=0.5):
                stop.wait(timeout=1.0)
                raise FatalReadError("done")
        obs = reader_observable(BlockingReader())
        disp = obs.subscribe(on_next=lambda x: None, on_error=lambda e: None)
        disp.dispose()
        assert stop.is_set()
class TestItems:
    def test_items_arrive_in_order(self):
        r = _FakeReader([10, 20, 30])
        items, errors, _ = collect(reader_observable(r))
        assert items == [10, 20, 30]
    def test_on_error_called_with_fatal(self):
        err = FatalReadError("fatal")
        r = _FakeReader([], terminal=err)
        items, errors, _ = collect(reader_observable(r))
        assert items == []
        assert len(errors) == 1
        assert errors[0] is err
class TestRetry:
    def test_retryable_error_is_not_immediately_fatal(self):
        """Single retryable error followed by success should not stop stream."""
        call_n = [0]
        class FlickyReader:
            def open(self): pass
            def close(self): pass
            def read(self, timeout=0.5):
                call_n[0] += 1
                if call_n[0] == 2:
                    raise RuntimeError("transient")
                if call_n[0] == 1 or call_n[0] == 3:
                    return call_n[0]
                raise FatalReadError("done")
        items, errors, _ = collect(reader_observable(FlickyReader(), max_consecutive_errors=5))
        assert 1 in items and 3 in items
    def test_max_consecutive_errors_terminates_stream(self):
        r = _FakeReader([], retryable_after=0)  # always raises RuntimeError
        items, errors, _ = collect(
            reader_observable(r, max_consecutive_errors=3)
        )
        assert items == []
        assert len(errors) == 1
        assert "3" in str(errors[0])
    def test_success_resets_consecutive_error_count(self):
        """Errors interspersed with successes should not accumulate to max."""
        call_n = [0]
        delivered = []
        class AlternatingReader:
            def open(self): pass
            def close(self): pass
            def read(self, timeout=0.5):
                call_n[0] += 1
                n = call_n[0]
                if n > 10:
                    raise FatalReadError("done")
                if n % 2 == 0:
                    raise RuntimeError("alternating error")
                return n
        items, errors, _ = collect(
            reader_observable(AlternatingReader(), max_consecutive_errors=2)
        )
        # Should receive multiple items (errors never reach 2 consecutive)
        assert len(items) >= 3
    def test_fatal_error_bypasses_retry_counter(self):
        """FatalReadError terminates immediately even with max_consecutive_errors=100."""
        r = _FakeReader([1, 2], terminal=FatalReadError("fatal"))
        items, errors, _ = collect(reader_observable(r, max_consecutive_errors=100))
        assert items == [1, 2]
        assert len(errors) == 1
        assert isinstance(errors[0], FatalReadError)
    def test_unlimited_retries_when_none(self):
        """max_consecutive_errors=None should retry more than 100 times."""
        call_n = [0]
        class EventuallyReader:
            def open(self): pass
            def close(self): pass
            def read(self, timeout=0.5):
                call_n[0] += 1
                if call_n[0] < 50:
                    raise RuntimeError("not yet")
                if call_n[0] == 50:
                    return "success"
                raise FatalReadError("done")
        items, errors, _ = collect(reader_observable(EventuallyReader(), max_consecutive_errors=None))
        assert "success" in items
