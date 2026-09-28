import unittest
from scripts.autoscale_european_checkers import decision, capacity_decision
class CheckerAutoscalingTests(unittest.TestCase):
    def test_scales_only_checker_bottleneck(self):
        self.assertEqual(decision(40,1,150,6,8,50,384,500),(4,'checker_bottleneck'))
        self.assertEqual(decision(72,6,280,44,10,90,384,500),(9,'checker_bottleneck'))
    def test_stops_for_source_parser_and_host_limits(self):
        cases=[((40,4,5,30,5,80,384,500),'parser_queue_not_full'),((40,4,150,4,5,80,384,500),'checker_cpu_not_saturated'),((40,4,150,30,35,80,384,500),'parser_capacity_limited'),((40,4,150,30,5,340,384,500),'cpu_headroom'),((40,4,150,30,5,80,384,10),'memory_headroom')]
        for args,reason in cases:self.assertEqual(decision(*args),(None,reason))

    def test_scales_parsers_only_when_busy_and_queue_depleted(self):
        self.assertEqual(capacity_decision(40,6,5,12,36,100,384,500,2),({'kind':'parser','instances':6,'parser_workers':60},'parser_bottleneck'))
        self.assertIsNone(capacity_decision(40,6,5,12,10,100,384,500,2)[0])
        self.assertEqual(capacity_decision(40,6,5,12,36,340,384,500,2),(None,'cpu_headroom'))
        self.assertEqual(capacity_decision(40,6,5,12,36,100,384,20,2),(None,'memory_headroom'))

    def test_full_queue_scales_checkers_even_when_parsers_are_busy(self):
        self.assertEqual(capacity_decision(40,4,150,30,36,100,384,500),({'kind':'checker','instances':6,'parser_workers':40},'checker_bottleneck'))
