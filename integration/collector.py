"""
Traffic Feature Collector

This module collects all real-time traffic parameters
from the running SUMO simulation using TraCI.

Current Features:
1. Traffic Flow
2. Vehicle Count

Upcoming Features:
3. Traffic Event Type (TEI)
4. Remaining Green Time
5. Downstream Occupancy
6. Queue Length
"""

import traci


class TrafficCollector:

    def __init__(self):
        """Initialize Traffic Collector"""
        pass

    # =====================================================
    # Vehicle Information
    # =====================================================

    def get_vehicle_ids(self):
        """
        Returns all active vehicle IDs.
        """
        return traci.vehicle.getIDList()

    def get_vehicle_count(self):
        """
        Returns total number of vehicles
        currently present in the network.
        """
        return len(self.get_vehicle_ids())

    # =====================================================
    # Traffic Flow
    # =====================================================

    def get_traffic_flow(self):
        """
        Returns total traffic flow detected by
        all induction loop detectors.
        """

        detector_ids = [
            "loop_NS",
            "loop_SN",
            "loop_EW",
            "loop_WE"
        ]

        total_flow = 0

        for detector in detector_ids:
            total_flow += traci.inductionloop.getLastStepVehicleNumber(detector)

        return total_flow

    # =====================================================
    # Feature Collector
    # =====================================================

    def collect_features(self):
        """
        Collect all available traffic features.

        Returns:
            dict
        """

        features = {

            "vehicle_count": self.get_vehicle_count(),

            "traffic_flow": self.get_traffic_flow()

        }

        return features