"""Reuse screening threads so thread-local checker sessions stay bounded.

Each batch still returns results in input order and drains all submitted work
before its context exits, including when a check fails. Only thread lifetime
changes; checker requests, diagnostics, cache keys and selection do not.
"""
from concurrent.futures import ThreadPoolExecutor, wait

_pools={}


class PersistentScreeningPool:
    def __init__(self, max_workers):
        if max_workers not in _pools:
            _pools[max_workers]=ThreadPoolExecutor(max_workers=max_workers)
        self.pool=_pools[max_workers]
        self.futures=[]

    def __enter__(self):
        return self

    def map(self, function, *iterables):
        self.futures.extend(self.pool.submit(function,*args) for args in zip(*iterables))
        return (future.result() for future in self.futures)

    def __exit__(self, *exc):
        wait(self.futures)
        self.futures.clear()
        return False


def close_screening_pools():
    for pool in _pools.values():
        pool.shutdown(wait=True)
    _pools.clear()
