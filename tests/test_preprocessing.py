import unittest

import numpy as np
import pandas as pd

from prediction.preprocessing import (
    StandardScaler,
    TrafficPreprocessor,
)


class TestPreprocessing(unittest.TestCase):

    def test_standard_scaler(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        self.assertEqual(scaled.shape, values.shape)
        restored = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(restored, values)

    def test_create_sequences(self):
        rows = []
        for t in range(6):
            rows.append({
                "location_id": "location_1",
                "junction_id": "junction_1",
                "simulation_time": float(t),
                "vehicle_count": 1,
                "traffic_flow": float(t + 1),
                "traffic_event_type": 0,
                "remaining_green_time": 10.0,
                "downstream_occupancy": 0.1 * t,
                "queue_length": float(t),
                "average_speed": 5.0,
                "waiting_time": 1.0,
                "current_signal_phase": 0,
                "downstream_queue_length": 0,
                "travel_time": 20.0,
                "arrival_rate": 1,
                "departure_rate": 1,
            })

        dataset = pd.DataFrame(rows)
        preprocessor = TrafficPreprocessor(lookback=3, prediction_horizon=2)
        X, y, metadata = preprocessor.create_sequences(dataset)

        self.assertEqual(X.ndim, 3)
        self.assertEqual(y.ndim, 3)
        self.assertEqual(len(metadata), X.shape[0])
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(y.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
