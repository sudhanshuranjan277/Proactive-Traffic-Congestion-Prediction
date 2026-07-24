"""
Traffic Data Pipeline

Responsible for:
- Validating collector output
- Maintaining fixed feature schema
- Aggregating 60-second traffic observations
- Generating dataset-ready junction rows
"""

import traci


class TrafficPipeline:

    FEATURE_SCHEMA = {
        "vehicle_count": int,
        "traffic_flow": int,
        "arrival_rate": int,
        "departure_rate": int,
        "traffic_event_type": int,
        "remaining_green_time": float,
        "current_signal_phase": int,
        "downstream_occupancy": float,
        "downstream_queue_length": int,
        "average_speed": float,
        "waiting_time": float,
        "travel_time": float,
        "queue_length": int,
    }

    def __init__(
        self,
        collector,
        location_id,
        observation_window=60,
    ):
        self.collector = collector
        self.location_id = location_id
        self.observation_window = observation_window
        self.step_count = 0
        self.window_data = {}

    def validate_features(self, junction_id, features):
        """
        Validate feature keys and data types.
        """
        expected_keys = set(self.FEATURE_SCHEMA.keys())
        received_keys = set(features.keys())

        if received_keys != expected_keys:
            missing_keys = expected_keys - received_keys
            extra_keys = received_keys - expected_keys
            raise ValueError(
                f"Feature schema mismatch for "
                f"junction {junction_id}. "
                f"Missing: {missing_keys}, "
                f"Extra: {extra_keys}"
            )

        for feature_name, expected_type in self.FEATURE_SCHEMA.items():
            value = features[feature_name]
            if type(value) is not expected_type:
                raise TypeError(
                    f"Invalid datatype for "
                    f"'{feature_name}' at junction "
                    f"{junction_id}. "
                    f"Expected {expected_type.__name__}, "
                    f"received {type(value).__name__}"
                )

    def initialize_window(self, junction_id):
        """
        Initialize aggregation storage for a junction.
        """
        self.window_data[junction_id] = {
            "traffic_flow": 0,
            "arrival_rate": 0,
            "departure_rate": 0,
            "occupancy_sum": 0.0,
            "average_speed_sum": 0.0,
            "waiting_time_sum": 0.0,
            "travel_time_sum": 0.0,
            "sample_count": 0,
        }

    def process_step(self):
        """
        Process one SUMO simulation step.

        Returns dataset-ready rows only when the observation window is complete.
        """
        junction_features = self.collector.collect_features()
        self.step_count += 1

        for junction_id, features in junction_features.items():
            self.validate_features(junction_id, features)
            if junction_id not in self.window_data:
                self.initialize_window(junction_id)

            window = self.window_data[junction_id]
            window["traffic_flow"] += features["traffic_flow"]
            window["arrival_rate"] += features["arrival_rate"]
            window["departure_rate"] += features["departure_rate"]
            window["occupancy_sum"] += features["downstream_occupancy"]
            window["average_speed_sum"] += features["average_speed"]
            window["waiting_time_sum"] += features["waiting_time"]
            window["travel_time_sum"] += features["travel_time"]
            window["sample_count"] += 1

        if self.step_count % self.observation_window != 0:
            return []

        dataset_rows = []
        simulation_time = float(traci.simulation.getTime())

        for junction_id, features in junction_features.items():
            window = self.window_data[junction_id]
            sample_count = max(window["sample_count"], 1)
            average_occupancy = window["occupancy_sum"] / sample_count
            average_speed = window["average_speed_sum"] / sample_count
            average_waiting_time = window["waiting_time_sum"] / sample_count
            average_travel_time = window["travel_time_sum"] / sample_count

            row = {
                "location_id": str(self.location_id),
                "junction_id": str(junction_id),
                "simulation_time": simulation_time,
                "vehicle_count": features["vehicle_count"],
                "traffic_flow": window["traffic_flow"],
                "arrival_rate": window["arrival_rate"],
                "departure_rate": window["departure_rate"],
                "traffic_event_type": features["traffic_event_type"],
                "remaining_green_time": features["remaining_green_time"],
                "current_signal_phase": features["current_signal_phase"],
                "downstream_occupancy": round(average_occupancy, 2),
                "downstream_queue_length": features["downstream_queue_length"],
                "average_speed": round(average_speed, 2),
                "waiting_time": round(average_waiting_time, 2),
                "travel_time": round(average_travel_time, 2),
                "queue_length": features["queue_length"],
            }
            dataset_rows.append(row)
            self.initialize_window(junction_id)

        return dataset_rows
