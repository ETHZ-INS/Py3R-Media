"""Unit tests for adaptive_pace operator."""

from __future__ import annotations

import statistics
import threading
import time

import pytest
import reactivex as rx
from reactivex.disposable import Disposable

from py3r.media.streaming.operators.adaptive_pace import adaptive_pace
from tests.unit.streaming.conftest import collect, from_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bursty_source(
    *,
    burst_size: int,
    bursts: int,
    period_s: float,
) -> tuple[rx.Observable[int], list[float]]:
    """
    Emits *burst_size* items as fast as possible, then waits so that one full
    cycle (burst + wait) takes *period_s* seconds. Repeats *bursts* times.

    Also records the wall-clock time of each on_next call into *source_times*.
    """
    source_times: list[float] = []

    def _subscribe(observer, _=None):
        stop = threading.Event()

        def run() -> None:
            value = 0
            for _ in range(bursts):
                burst_start = time.monotonic()
                for _ in range(burst_size):
                    if stop.is_set():
                        return
                    source_times.append(time.monotonic())
                    observer.on_next(value)
                    value += 1

                remaining = period_s - (time.monotonic() - burst_start)
                if remaining > 0 and stop.wait(timeout=remaining):
                    return

            observer.on_completed()

        producer = threading.Thread(target=run, daemon=True)
        producer.start()

        def _dispose() -> None:
            stop.set()
            if threading.current_thread() is not producer:
                producer.join(timeout=1.0)

        return Disposable(_dispose)

    return rx.create(_subscribe), source_times


def collect_timed(
    source: rx.Observable[int],
    timeout: float = 8.0,
) -> tuple[list[int], list[float], list[BaseException]]:
    """Like conftest.collect but also records per-item wall-clock timestamps."""
    items: list[int] = []
    out_times: list[float] = []
    errors: list[BaseException] = []
    done = threading.Event()

    source.subscribe(
        on_next=lambda x: (items.append(x), out_times.append(time.monotonic())),
        on_error=lambda e: (errors.append(e), done.set()),
        on_completed=done.set,
    )
    done.wait(timeout=timeout)
    return items, out_times, errors


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptivePaceValidation:
    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            adaptive_pace(window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            adaptive_pace(window_size=-5)

    def test_initial_interval_negative_raises(self):
        with pytest.raises(ValueError, match="initial_interval"):
            adaptive_pace(initial_interval=-0.033)

    def test_min_interval_zero_raises(self):
        with pytest.raises(ValueError, match="min_interval"):
            adaptive_pace(min_interval=0.0)

    def test_min_interval_negative_raises(self):
        with pytest.raises(ValueError, match="min_interval"):
            adaptive_pace(min_interval=-1.0)

    def test_valid_defaults_do_not_raise(self):
        adaptive_pace()  # should not raise

    def test_valid_explicit_args_do_not_raise(self):
        adaptive_pace(window_size=16, initial_interval=0.033, min_interval=1e-3)


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestAdaptivePaceCorrectness:
    def test_preserves_order_and_no_drop(self):
        """Every item emitted by upstream arrives downstream, in order."""
        source, _ = make_bursty_source(burst_size=6, bursts=4, period_s=0.6)

        items, _, errors = collect_timed(source.pipe(adaptive_pace()))

        assert errors == []
        assert items == list(range(24))

    def test_empty_source_completes(self):
        items, errors, _ = collect(rx.empty().pipe(adaptive_pace()), timeout=2.0)

        assert errors == []
        assert items == []

    def test_single_item_delivered_and_completes(self):
        items, errors, _ = collect(rx.of(42).pipe(adaptive_pace()), timeout=2.0)

        assert errors == []
        assert items == [42]

    def test_multiple_items_synchronous_source(self):
        items, errors, _ = collect(from_items(1, 2, 3).pipe(adaptive_pace()), timeout=4.0)

        assert errors == []
        assert items == [1, 2, 3]

    def test_error_propagated(self):
        sentinel = RuntimeError("boom")
        items, errors, _ = collect(
            from_items(error=sentinel).pipe(adaptive_pace()),
            timeout=2.0,
        )

        assert errors == [sentinel]
        assert items == []

    def test_error_delivered_after_buffered_items(self):
        """
        The error sentinel is placed in the queue *after* on_next items, so all
        buffered items must arrive before the error is forwarded downstream.
        """
        sentinel = ValueError("upstream error")
        items: list[int] = []
        errors: list[BaseException] = []
        done = threading.Event()

        from_items(0, 1, 2, 3, 4, error=sentinel).pipe(
            adaptive_pace(initial_interval=0.01, window_size=4)
        ).subscribe(
            on_next=items.append,
            on_error=lambda e: (errors.append(e), done.set()),
            on_completed=done.set,
        )

        done.wait(timeout=4.0)
        assert errors == [sentinel]
        assert items == [0, 1, 2, 3, 4]

    def test_completed_delivered_after_buffered_items(self):
        """on_completed is queued after on_next items and must arrive last."""
        items: list[int] = []
        errors: list[BaseException] = []
        done = threading.Event()

        from_items(0, 1, 2, 3, 4, 5).pipe(
            adaptive_pace(initial_interval=0.01, window_size=4)
        ).subscribe(
            on_next=items.append,
            on_error=lambda e: (errors.append(e), done.set()),
            on_completed=done.set,
        )

        done.wait(timeout=4.0)
        assert errors == []
        assert items == [0, 1, 2, 3, 4, 5]

    def test_emissions_on_worker_thread(self):
        """Items must be emitted from the operator's internal worker thread."""
        main_thread = threading.current_thread()
        emit_threads: list[threading.Thread] = []
        done = threading.Event()

        rx.of(1, 2, 3).pipe(adaptive_pace()).subscribe(
            on_next=lambda _: emit_threads.append(threading.current_thread()),
            on_completed=done.set,
        )
        done.wait(timeout=2.0)

        assert len(emit_threads) == 3
        assert all(t is not main_thread for t in emit_threads)

    def test_worker_thread_is_named(self):
        """The worker thread name should identify this operator."""
        observed_name: list[str] = []
        done = threading.Event()

        rx.of(1).pipe(adaptive_pace()).subscribe(
            on_next=lambda _: observed_name.append(threading.current_thread().name),
            on_completed=done.set,
        )
        done.wait(timeout=2.0)

        assert observed_name and "adaptive_pace" in observed_name[0]


# ---------------------------------------------------------------------------
# Pacing / smoothing behaviour
# ---------------------------------------------------------------------------

class TestAdaptivePacePacing:
    def test_smooths_bursty_cadence(self):
        """
        Output inter-item jitter (pstdev) must be substantially less than input
        jitter after a short warmup, and the inferred mean pace should stay near
        the true frame rate.
        """
        source, source_times = make_bursty_source(burst_size=6, bursts=5, period_s=0.6)

        items, out_times, errors = collect_timed(source.pipe(adaptive_pace()))

        assert errors == []
        assert len(items) == 30

        in_deltas = [t2 - t1 for t1, t2 in zip(source_times, source_times[1:])]
        out_deltas = [t2 - t1 for t1, t2 in zip(out_times, out_times[1:])]

        # Skip startup transients while the estimator learns a baseline.
        warmup = 8
        in_tail = in_deltas[warmup:]
        out_tail = out_deltas[warmup:]

        assert len(in_tail) > 5
        assert len(out_tail) > 5

        in_std = statistics.pstdev(in_tail)
        out_std = statistics.pstdev(out_tail)
        out_mean = statistics.fmean(out_tail)

        # Output should be materially less bursty than input.
        assert out_std < in_std * 0.6, (
            f"Output not smooth enough: in_std={in_std:.4f}, out_std={out_std:.4f}"
        )
        # burst_size / period_s = 10 fps → ~0.1 s per frame.
        assert 0.05 < out_mean < 0.18, f"Mean pace out of range: {out_mean:.4f} s"

    def test_initial_interval_primes_estimator(self):
        """
        With initial_interval set the estimator window is pre-seeded, so pacing
        is active from the very first emission without needing a warmup burst.
        """
        target = 0.05  # 20 fps
        source, _ = make_bursty_source(burst_size=10, bursts=3, period_s=0.5)

        _, out_times, errors = collect_timed(
            source.pipe(adaptive_pace(initial_interval=target, window_size=32)),
            timeout=6.0,
        )

        assert errors == []
        out_deltas = [t2 - t1 for t1, t2 in zip(out_times, out_times[1:])]

        # Pre-seeded estimator → smaller warmup needed.
        warmup = 3
        tail = out_deltas[warmup:]
        assert len(tail) > 5

        mean = statistics.fmean(tail)
        # Should stay near the seeded hint (within ~3× to tolerate adaptation).
        assert 0.02 < mean < 0.15, (
            f"Mean pace unexpected with initial_interval={target}: {mean:.4f} s"
        )

    def test_initial_interval_zero_clamped_to_min_interval(self):
        """initial_interval=0 is a valid argument; it gets clamped to min_interval."""
        items, errors, _ = collect(
            from_items(1, 2, 3).pipe(
                adaptive_pace(initial_interval=0.0, min_interval=0.01)
            ),
            timeout=2.0,
        )

        assert errors == []
        assert items == [1, 2, 3]

    def test_min_interval_floors_pace(self):
        """
        When items arrive faster than min_interval the inter-emit times should be
        at least ~min_interval (with a loose tolerance for OS scheduling noise).
        """
        min_iv = 0.05

        _, out_times, errors = collect_timed(
            from_items(*range(8)).pipe(
                adaptive_pace(min_interval=min_iv, window_size=8)
            ),
            timeout=5.0,
        )

        assert errors == []
        out_deltas = [t2 - t1 for t1, t2 in zip(out_times, out_times[1:])]

        for dt in out_deltas:
            assert dt >= min_iv * 0.5, (
                f"Inter-emit gap {dt:.4f} s is below min_interval {min_iv} s"
            )

    def test_window_size_one_still_works(self):
        """window_size=1 keeps only the last interval; operator should still complete."""
        source, _ = make_bursty_source(burst_size=5, bursts=3, period_s=0.4)

        items, _, errors = collect_timed(source.pipe(adaptive_pace(window_size=1)))

        assert errors == []
        assert items == list(range(15))

    def test_no_catch_up_debt_after_gap(self):
        """
        After a pause in upstream, ``next_due = max(next_due + p, now)`` prevents
        the operator from building an infinite debt and emitting the next batch as
        a burst.  Items after the gap should resume at the normal pace.
        """
        emit_times: list[float] = []
        done = threading.Event()

        def _subscribe(observer, _=None):
            def run():
                for i in range(8):
                    observer.on_next(i)
                time.sleep(0.6)
                for i in range(8, 16):
                    observer.on_next(i)
                observer.on_completed()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            return Disposable()

        rx.create(_subscribe).pipe(
            adaptive_pace(initial_interval=0.04, window_size=8)
        ).subscribe(
            on_next=lambda _: emit_times.append(time.monotonic()),
            on_completed=done.set,
        )

        done.wait(timeout=8.0)
        assert len(emit_times) == 16

        # The very first item after the gap (index 8 → 9 delta) may fire
        # immediately (next_due reset), so check deltas from item 9 onward.
        second_batch_inner_deltas = [
            t2 - t1 for t1, t2 in zip(emit_times[9:], emit_times[10:])
        ]
        assert len(second_batch_inner_deltas) > 0

        for dt in second_batch_inner_deltas:
            assert dt >= 0.02, (
                f"Catch-up burst detected after gap: {dt:.4f} s"
            )


# ---------------------------------------------------------------------------
# Disposal
# ---------------------------------------------------------------------------

class TestAdaptivePaceDisposal:
    def test_dispose_stops_emission(self):
        """Disposing mid-stream stops further emissions."""
        items: list[int] = []
        done = threading.Event()

        def _subscribe(observer, _=None):
            stop = threading.Event()

            def run():
                for i in range(50):
                    if stop.is_set():
                        return
                    observer.on_next(i)
                    stop.wait(timeout=0.1)
                observer.on_completed()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            return Disposable(lambda: (stop.set(), t.join(timeout=1.0)))

        disp = rx.create(_subscribe).pipe(
            adaptive_pace(initial_interval=0.05)
        ).subscribe(
            on_next=items.append,
            on_completed=done.set,
        )

        time.sleep(0.4)
        disp.dispose()
        snapshot = len(items)

        time.sleep(0.4)

        assert len(items) == snapshot, "Emissions continued after dispose()"
        assert not done.is_set(), "on_completed should not fire after dispose()"

    def test_dispose_before_any_emission(self):
        """Disposing before any item arrives must not raise or deadlock."""
        done = threading.Event()

        def _subscribe(observer, _=None):
            def run():
                time.sleep(0.5)
                observer.on_next(1)
                observer.on_completed()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            return Disposable()

        disp = rx.create(_subscribe).pipe(adaptive_pace()).subscribe(
            on_next=lambda _: None,
            on_completed=done.set,
        )

        disp.dispose()  # dispose before the item arrives
        time.sleep(0.7)
        # Test passes if nothing raises and the process is not stuck.
