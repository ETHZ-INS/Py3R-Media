import threading
import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.scheduler import EventLoopScheduler

from py3r.media.video import VideoSource


def video_source_observable(src: VideoSource, scheduler: EventLoopScheduler):
    """
    Create an Observable that:
      - opens the VideoSource on subscribe (on `scheduler`)
      - reads frames on `scheduler` via a recursive scheduled action
      - closes the VideoSource on dispose/completion (on `scheduler`)
    """

    def resource_factory():
        # open on the scheduler thread
        print(f"_open {threading.current_thread().name}")
        src.open()

        # close on the scheduler thread (even if dispose() is called elsewhere)
        def _dispose():
            print(f"_dispose {threading.current_thread().name}")
            src.close()
        return Disposable(_dispose)

    def observable_factory(_res):
        def _subscribe(observer, __):
            cancelled = [False]

            def tick(_, __=None):
                if cancelled[0]:
                    return

                try:
                    frame = src.read(timeout=0.5)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    observer.on_error(e)
                    return

                if frame is None:
                    # EOF/no frame
                    observer.on_completed()
                    return

                observer.on_next(frame)
                scheduler.schedule(tick)

            # start the loop on the scheduler
            scheduler.schedule(tick)

            # cooperative cancellation; close happens via the resource's dispose()
            def _cancel():
                cancelled[0] = True

            return Disposable(_cancel)

        # Note: _subscribe ignores incoming scheduler and uses our chosen one
        return rx.create(_subscribe)

    # Wrap open/close in `using`, and ensure subscribe/dispose are marshalled
    return rx.using(resource_factory, observable_factory).pipe(
        ops.subscribe_on(scheduler)   # subscription & teardown scheduled on `scheduler`
    )