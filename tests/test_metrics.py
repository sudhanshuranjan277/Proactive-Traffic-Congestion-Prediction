import unittest

import pandas as pd

from evaluation.metrics import summarize_metrics


class TestMetrics(unittest.TestCase):

    def test_summarize_metrics(self):
        dataset = pd.DataFrame(
            [
                {
                    "queue_length": 5,
                    "waiting_time": 10.0,
                    "average_speed": 15.0,
                    "departure_rate": 4,
                    "downstream_occupancy": 0.95,
                    "traffic_event_type": 2,
                },
                {
                    "queue_length": 3,
                    "waiting_time": 8.0,
                    "average_speed": 20.0,
                    "departure_rate": 5,
                    "downstream_occupancy": 0.75,
                    "traffic_event_type": 0,
                },
            ]
        )
        metrics = summarize_metrics(dataset)
        self.assertAlmostEqual(metrics["average_queue_length"], 4.0)
        self.assertAlmostEqual(metrics["average_waiting_time"], 9.0)
        self.assertAlmostEqual(metrics["average_speed"], 17.5)
        self.assertAlmostEqual(metrics["total_throughput"], 9.0)
        self.assertEqual(metrics["spillback_events"], 1)
        self.assertAlmostEqual(metrics["congestion_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
