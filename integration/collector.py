"""
Traffic Feature Collector

Collects real-time traffic parameters
from SUMO using TraCI.

Current Features:
1. Traffic Flow
2. Vehicle Count

Planned Features:
3. Traffic Event Type
4. Remaining Green Time
5. Downstream Occupancy
6. Queue Length
"""

import traci


class TrafficCollector:

    def __init__(self):

        self.traffic_detectors = [
            "loop_osm_1",
            "loop_osm_2",
        ]

    def get_vehicle_ids(self):
        """
        Return all active vehicle IDs.
        """

        return traci.vehicle.getIDList()

    def get_vehicle_count(self):
        """
        Return total number of active vehicles.
        """

        return len(self.get_vehicle_ids())

    def get_traffic_flow(self):
        """
        Calculate traffic flow using
        OSM induction loop detectors.
        """

        total_flow = 0

        available_detectors = set(
            traci.inductionloop.getIDList()
        )

        for detector in self.traffic_detectors:

            if detector in available_detectors:

                vehicle_count = (
                    traci.inductionloop
                    .getLastStepVehicleNumber(detector)
                )

                total_flow += vehicle_count

        return total_flow

    def collect_features(self):
        """
        Collect all currently implemented
        real-time traffic features.
        """

        return {
            "vehicle_count": self.get_vehicle_count(),
            "traffic_flow": self.get_traffic_flow(),
        }