import multiprocessing
import unittest
from scripts.worker_pool import PrestartedProcessPool


def square(value):
    return value*value


class WorkerPoolTests(unittest.TestCase):
    def test_full_pool_exists_before_generation_and_returns_same_results(self):
        previous={p.pid for p in multiprocessing.active_children()}
        with PrestartedProcessPool(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
            current={p.pid for p in multiprocessing.active_children()}
            self.assertEqual(len(current-previous),4)
            self.assertEqual(list(pool.map(square,range(12))),[x*x for x in range(12)])
        self.assertEqual({p.pid for p in multiprocessing.active_children()},previous)
