from concurrent.futures import Future
from reactivex import Observer
import threading

class NotifyingObserver(Observer):
    def __init__(self, inner: Observer):
        super().__init__()
        self.inner = inner
        self.result = Future()
        self._lock = threading.Lock()

    def _set_result_ok(self):
        with self._lock:
            if not self.result.done():
                self.result.set_result(None)

    def _set_result_err(self, e: Exception):
        with self._lock:
            if not self.result.done():
                self.result.set_exception(e)

    def on_next(self, value):
        try:
            self.inner.on_next(value)
        except Exception as e:
            # Translate inner observer bug into an error result
            try:
                self.inner.on_error(e)
            finally:
                self._set_result_err(e)

    def on_error(self, err: Exception):
        try:
            self.inner.on_error(err)
        finally:
            self._set_result_err(err)
        super().on_error(err)  # sets is_stopped

    def on_completed(self):
        try:
            self.inner.on_completed()
        finally:
            self._set_result_ok()
        super().on_completed()  # sets is_stopped
