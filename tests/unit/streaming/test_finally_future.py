"""Unit tests for finally_future operator."""
from __future__ import annotations
from concurrent.futures import Future
import reactivex as rx
from reactivex.disposable import Disposable
from py3r.media.streaming.operators.finally_future import finally_future
from tests.unit.streaming.conftest import collect, from_items
class TestCompleted:
    def test_future_resolved_on_completed(self):
        f = Future()
        collect(from_items(1, 2).pipe(finally_future(f)))
        assert f.done() and f.result() is None
    def test_items_pass_through(self):
        f = Future()
        items, errors, _ = collect(from_items(10, 20).pipe(finally_future(f)))
        assert items == [10, 20] and errors == []
    def test_future_resolved_only_once(self):
        f = Future()
        _, _, disp = collect(from_items(1).pipe(finally_future(f)))
        disp.dispose()
        assert f.done() and f.result() is None
class TestError:
    def test_future_gets_exception_on_error(self):
        f = Future()
        err = RuntimeError("source failed")
        collect(from_items(error=err).pipe(finally_future(f)))
        assert f.done() and f.exception() is err
    def test_error_passes_through_to_observer(self):
        f = Future()
        err = ValueError("x")
        _, errors, _ = collect(from_items(error=err).pipe(finally_future(f)))
        assert errors[0] is err
class TestDispose:
    def test_cancel_on_dispose_true_cancels_future(self):
        f = Future()
        disp = rx.never().pipe(finally_future(f, cancel_on_dispose=True)).subscribe()
        disp.dispose()
        assert f.cancelled()
    def test_cancel_on_dispose_false_resolves_future(self):
        f = Future()
        disp = rx.never().pipe(finally_future(f, cancel_on_dispose=False)).subscribe()
        disp.dispose()
        assert f.done() and not f.cancelled() and f.result() is None
    def test_double_dispose_is_safe(self):
        f = Future()
        disp = rx.never().pipe(finally_future(f, cancel_on_dispose=True)).subscribe()
        disp.dispose()
        disp.dispose()
    def test_dispose_after_complete_keeps_result(self):
        f = Future()
        _, _, disp = collect(from_items(1).pipe(finally_future(f, cancel_on_dispose=True)))
        assert f.result() is None
        disp.dispose()
        assert not f.cancelled() and f.result() is None
class TestOnNextException:
    def test_on_next_exception_sets_future(self):
        f = Future()
        err = ValueError("downstream misbehaved")
        def misbehaving_source_sub(observer, _=None):
            try:
                observer.on_next(42)
            except Exception:
                pass
            observer.on_completed()
            return Disposable()
        source = rx.create(misbehaving_source_sub)
        def bad_next(x):
            raise err
        source.pipe(finally_future(f)).subscribe(on_next=bad_next, on_error=lambda e: None)
        assert f.done() and f.exception() is err
