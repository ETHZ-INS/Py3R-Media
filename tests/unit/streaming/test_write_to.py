"""
Unit tests for write_to operator.
"""
from __future__ import annotations

import reactivex as rx
from unittest.mock import MagicMock, call

from py3r.media.streaming.operators.write_to import write_to
from tests.unit.streaming.conftest import collect, from_items


class TestLifecycle:
    def test_open_called_on_subscribe(self):
        writer = MagicMock()
        collect(from_items(1, 2).pipe(write_to(writer)))
        writer.open.assert_called_once()

    def test_close_called_on_completed(self):
        writer = MagicMock()
        collect(from_items(1, 2).pipe(write_to(writer)))
        writer.close.assert_called_once()

    def test_close_called_on_error(self):
        writer = MagicMock()
        err = RuntimeError("upstream error")
        collect(from_items(1, error=err).pipe(write_to(writer)))
        writer.close.assert_called_once()

    def test_close_called_on_dispose(self):
        writer = MagicMock()
        disp = rx.never().pipe(write_to(writer)).subscribe()
        disp.dispose()
        writer.close.assert_called_once()

    def test_close_called_exactly_once_complete_then_dispose(self):
        writer = MagicMock()
        _, _, disp = collect(from_items(1).pipe(write_to(writer)))
        disp.dispose()  # second cleanup after already completed
        writer.close.assert_called_once()

    def test_close_called_exactly_once_error_then_dispose(self):
        writer = MagicMock()
        err = RuntimeError("x")
        _, _, disp = collect(from_items(error=err).pipe(write_to(writer)))
        disp.dispose()
        writer.close.assert_called_once()

    def test_open_called_before_any_write(self):
        call_log = []
        writer = MagicMock()
        writer.open.side_effect = lambda: call_log.append("open")
        writer.write.side_effect = lambda x: call_log.append(f"write({x})")
        collect(from_items(1).pipe(write_to(writer)))
        assert call_log[0] == "open"
        assert call_log[1] == "write(1)"


class TestWrite:
    def test_write_called_for_each_item(self):
        writer = MagicMock()
        collect(from_items(10, 20, 30).pipe(write_to(writer)))
        assert writer.write.call_count == 3
        writer.write.assert_has_calls([call(10), call(20), call(30)])

    def test_write_not_called_after_close(self):
        """Dispose before any items → write never called."""
        writer = MagicMock()
        disp = rx.never().pipe(write_to(writer)).subscribe()
        disp.dispose()
        writer.write.assert_not_called()

    def test_items_pass_through_to_downstream(self):
        writer = MagicMock()
        items, errors, _ = collect(from_items(1, 2, 3).pipe(write_to(writer)))
        assert items == [1, 2, 3]
        assert errors == []

    def test_error_passes_through_to_downstream(self):
        writer = MagicMock()
        err = ValueError("oops")
        items, errors, _ = collect(from_items(1, error=err).pipe(write_to(writer)))
        assert items == [1]
        assert len(errors) == 1
        assert errors[0] is err

    def test_write_receives_correct_types(self):
        writer = MagicMock()
        values = ["hello", 42, None, {"key": "val"}]
        collect(from_items(*values).pipe(write_to(writer)))
        for i, v in enumerate(values):
            assert writer.write.call_args_list[i] == call(v)

    def test_empty_source_no_writes(self):
        writer = MagicMock()
        collect(from_items().pipe(write_to(writer)))
        writer.write.assert_not_called()
        writer.open.assert_called_once()
        writer.close.assert_called_once()

