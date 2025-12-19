import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import CurrentThreadScheduler

from py3r.media.types import VideoFrame
from py3r.media.video import VideoSource


def video_source_observable(src: VideoSource, scheduler: rx.abc.SchedulerBase = None) -> rx.Observable[VideoFrame]:
    """
    Create an Observable that:
      - opens the VideoSource on subscribe (on `scheduler`)
      - reads frames on `scheduler` via a recursive scheduled action
      - closes the VideoSource on dispose/completion (on `scheduler`)
    """

    def resource_factory():
        # open on the scheduler thread
        src.open()

        # close on the scheduler thread (even if dispose() is called elsewhere)
        def _dispose():
            src.close()
        return Disposable(_dispose)

    def observable_factory(_res):
        def _subscribe(observer, scheduler_=None):
            _scheduler = scheduler or scheduler_ or CurrentThreadScheduler.singleton()
            cancelled = [False]

            def tick(_, __=None):
                if cancelled[0]:
                    return

                try:
                    frame = src.read(timeout=5.0)
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
                _scheduler.schedule(tick)

            # start the loop on the scheduler
            _scheduler.schedule(tick)

            # cooperative cancellation; close happens via the resource's dispose()
            def _cancel():
                cancelled[0] = True

            return Disposable(_cancel)

        # Note: _subscribe ignores incoming scheduler and uses our chosen one
        return rx.create(_subscribe)

    return rx.using(resource_factory, observable_factory)#.pipe(ops.subscribe_on(scheduler))
