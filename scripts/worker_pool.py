"""Start every configured parser process, including on checkpoint-heavy resumes.

Use public executor APIs. The event prevents completed startup tasks from being
mistaken for idle workers while the initial process allocation is still growing.
The original parser initializer and all submitted generation tasks are unchanged.
"""
from concurrent.futures import ProcessPoolExecutor


def initialize_and_wait(initializer, initargs, ready):
    if initializer is not None:
        initializer(*initargs)
    ready.wait()


def ready_task():
    return True


class PrestartedProcessPool(ProcessPoolExecutor):
    def __init__(self, *, max_workers, mp_context, initializer=None, initargs=()):
        self.worker_count=max_workers
        self.startup_ready=mp_context.Event()
        super().__init__(max_workers=max_workers, mp_context=mp_context,
                         initializer=initialize_and_wait,
                         initargs=(initializer, initargs, self.startup_ready))

    def __enter__(self):
        super().__enter__()
        try:
            startup=[self.submit(ready_task) for _ in range(self.worker_count)]
        finally:
            self.startup_ready.set()
        for future in startup:
            future.result()
        return self
