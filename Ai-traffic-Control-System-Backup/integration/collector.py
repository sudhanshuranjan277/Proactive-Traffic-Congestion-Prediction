"""
Dynamic Multi-Junction Traffic Feature Collector

Collects junction-wise real-time traffic parameters
from SUMO using TraCI.

Current Features:
1. Traffic Flow
2. Vehicle Count
3. Traffic Event Type
4. Remaining Green Time
5. Downstream Occupancy
6. Queue Length
7. Average Speed
8. Waiting Time
9. Current Signal Phase
10. Downstream Queue Length
11. Travel Time
12. Arrival Rate
13. Departure Rate
"""

import traci

from config import (
    COLLECTOR_STOPPED_SPEED_THRESHOLD,
    COLLECTOR_CONGESTION_STOPPED_VEHICLES,
    COLLECTOR_CONGESTION_SPEED_THRESHOLD,
    COLLECTOR_SLOW_TRAFFIC_SPEED_THRESHOLD,
    COLLECTOR_ROUNDING_PRECISION,
)


class TrafficCollector:

    # Traffic event classification codes.
    EVENT_NORMAL = 0
    EVENT_SLOW_TRAFFIC = 1
    EVENT_CONGESTION = 2

    def __init__(
        self,
        stopped_speed_threshold=None,
        congestion_stopped_vehicles=None,
        congestion_speed_threshold=None,
        slow_traffic_speed_threshold=None,
        rounding_precision=None,
    ):

        self.junction_lanes = {}
        self.junction_downstream_lanes = {}
        self.previous_lane_vehicles = {}
        self.initialized = False

        self.stopped_speed_threshold = (
            stopped_speed_threshold
            if stopped_speed_threshold is not None
            else COLLECTOR_STOPPED_SPEED_THRESHOLD
        )

        self.congestion_stopped_vehicles = (
            congestion_stopped_vehicles
            if congestion_stopped_vehicles is not None
            else COLLECTOR_CONGESTION_STOPPED_VEHICLES
        )

        self.congestion_speed_threshold = (
            congestion_speed_threshold
            if congestion_speed_threshold is not None
            else COLLECTOR_CONGESTION_SPEED_THRESHOLD
        )

        self.slow_traffic_speed_threshold = (
            slow_traffic_speed_threshold
            if slow_traffic_speed_threshold is not None
            else COLLECTOR_SLOW_TRAFFIC_SPEED_THRESHOLD
        )

        self.rounding_precision = (
            rounding_precision
            if rounding_precision is not None
            else COLLECTOR_ROUNDING_PRECISION
        )

    def initialize_junctions(self):
        """
        Discover signalized junctions,
        controlled incoming lanes,
        and downstream lanes.
        """

        traffic_light_ids = (
            traci.trafficlight.getIDList()
        )

        for junction_id in traffic_light_ids:

            controlled_lanes = (
                traci.trafficlight.getControlledLanes(
                    junction_id
                )
            )

            unique_lanes = list(
                dict.fromkeys(controlled_lanes)
            )

            self.junction_lanes[
                junction_id
            ] = unique_lanes

            downstream_lanes = set()

            controlled_links = (
                traci.trafficlight.getControlledLinks(
                    junction_id
                )
            )

            for signal_links in controlled_links:

                for link in signal_links:

                    if len(link) >= 2:

                        downstream_lane = link[1]

                        if downstream_lane:

                            downstream_lanes.add(
                                downstream_lane
                            )

            self.junction_downstream_lanes[
                junction_id
            ] = list(
                downstream_lanes
            )

            for lane_id in unique_lanes:

                self.previous_lane_vehicles[
                    lane_id
                ] = set()

        self.initialized = True

        print(
            f"Discovered Signalized Junctions: "
            f"{len(self.junction_lanes)}"
        )

    def get_lane_vehicle_ids(
        self,
        lane_id
    ):
        """
        Return vehicle IDs currently
        present on a lane.
        """

        return set(
            traci.lane.getLastStepVehicleIDs(
                lane_id
            )
        )

    def get_junction_vehicle_ids(
        self,
        junction_id
    ):
        """
        Return unique vehicles present
        on controlled incoming lanes.
        """

        vehicle_ids = set()

        for lane_id in self.junction_lanes[
            junction_id
        ]:

            vehicle_ids.update(
                self.get_lane_vehicle_ids(
                    lane_id
                )
            )

        return vehicle_ids

    def get_vehicle_count(
        self,
        junction_id
    ):
        """
        Return junction-specific
        vehicle count.
        """

        return len(
            self.get_junction_vehicle_ids(
                junction_id
            )
        )

    def get_flow_rates(
        self,
        junction_id
    ):
        """
        Return vehicle arrival and departure
        counts for controlled incoming lanes
        compared with the previous simulation step.
        """

        total_arrival = 0
        total_departure = 0

        for lane_id in self.junction_lanes[
            junction_id
        ]:

            current_vehicles = (
                self.get_lane_vehicle_ids(
                    lane_id
                )
            )

            previous_vehicles = (
                self.previous_lane_vehicles.get(
                    lane_id,
                    set(),
                )
            )

            total_arrival += len(
                current_vehicles
                - previous_vehicles
            )

            total_departure += len(
                previous_vehicles
                - current_vehicles
            )

            self.previous_lane_vehicles[
                lane_id
            ] = current_vehicles

        return (
            int(total_arrival),
            int(total_departure),
        )

    def get_traffic_flow(
        self,
        junction_id
    ):
        """
        Calculate newly arriving vehicles
        on controlled incoming lanes.

        This helper updates lane vehicle state.
        It should not be called together with
        other flow-rate helpers in the same
        simulation step.
        """

        arrival, _ = self.get_flow_rates(
            junction_id
        )

        return arrival

    def get_arrival_rate(
        self,
        junction_id
    ):
        """
        Return vehicles entering controlled
        incoming lanes compared with the
        previous simulation step.

        This helper updates lane vehicle state.
        """

        arrival, _ = self.get_flow_rates(
            junction_id
        )

        return arrival

    def get_departure_rate(
        self,
        junction_id
    ):
        """
        Return vehicles leaving controlled
        incoming lanes compared with the
        previous simulation step.

        This helper updates lane vehicle state.
        """

        _, departure = self.get_flow_rates(
            junction_id
        )

        return departure

    def get_average_speed(
        self,
        junction_id
    ):
        """
        Return average speed of vehicles
        currently present on controlled
        incoming lanes.
        """

        vehicle_ids = (
            self.get_junction_vehicle_ids(
                junction_id
            )
        )

        if not vehicle_ids:
            return 0.0

        speeds = [
            traci.vehicle.getSpeed(
                vehicle_id
            )
            for vehicle_id in vehicle_ids
        ]

        average_speed = (
            sum(speeds)
            / len(speeds)
        )

        return round(
            float(average_speed),
            self.rounding_precision,
        )

    def get_waiting_time(
        self,
        junction_id
    ):
        """
        Return average lane waiting time
        across controlled incoming lanes.
        """

        waiting_times = []

        for lane_id in self.junction_lanes[
            junction_id
        ]:

            waiting_times.append(
                traci.lane.getWaitingTime(
                    lane_id
                )
            )

        average_waiting_time = (
            sum(waiting_times)
            / len(waiting_times)
            if waiting_times
            else 0.0
        )

        return round(
            float(average_waiting_time),
            self.rounding_precision,
        )

    def get_current_signal_phase(
        self,
        junction_id
    ):
        """
        Return current traffic-light
        phase index.
        """

        return int(
            traci.trafficlight.getPhase(
                junction_id
            )
        )

    def get_remaining_green_time(
        self,
        junction_id
    ):
        """
        Calculate remaining time
        of the active green phase.
        """

        signal_state = (
            traci.trafficlight
            .getRedYellowGreenState(
                junction_id
            )
        )

        if (
            "G" not in signal_state
            and "g" not in signal_state
        ):
            return 0.0

        next_switch = (
            traci.trafficlight.getNextSwitch(
                junction_id
            )
        )

        simulation_time = (
            traci.simulation.getTime()
        )

        remaining_time = max(
            0.0,
            next_switch - simulation_time,
        )

        return round(
            float(remaining_time),
            self.rounding_precision,
        )

    def get_downstream_occupancy(
        self,
        junction_id
    ):
        """
        Calculate average downstream
        lane occupancy percentage.
        """

        downstream_lanes = (
            self.junction_downstream_lanes.get(
                junction_id,
                [],
            )
        )

        if not downstream_lanes:
            return 0.0

        occupancies = [
            traci.lane.getLastStepOccupancy(
                lane_id
            )
            for lane_id in downstream_lanes
        ]

        average_occupancy = (
            sum(occupancies)
            / len(occupancies)
        )

        return round(
            float(average_occupancy),
            self.rounding_precision,
        )

    def get_downstream_queue_length(
        self,
        junction_id
    ):
        """
        Calculate halted vehicles
        on downstream lanes.
        """

        total_queue = 0

        for lane_id in (
            self.junction_downstream_lanes.get(
                junction_id,
                [],
            )
        ):

            total_queue += (
                traci.lane
                .getLastStepHaltingNumber(
                    lane_id
                )
            )

        return int(total_queue)

    def get_travel_time(
        self,
        junction_id
    ):
        """
        Calculate average accumulated trip time
        of vehicles currently present on
        controlled incoming lanes.

        Vehicle travel time is calculated from
        the current simulation time and the
        vehicle's actual departure time.
        """

        vehicle_ids = (
            self.get_junction_vehicle_ids(
                junction_id
            )
        )

        if not vehicle_ids:
            return 0.0

        simulation_time = (
            traci.simulation.getTime()
        )

        travel_times = []

        for vehicle_id in vehicle_ids:

            departure_time = (
                traci.vehicle.getDeparture(
                    vehicle_id
                )
            )

            travel_time = max(
                0.0,
                simulation_time - departure_time,
            )

            travel_times.append(
                travel_time
            )

        average_travel_time = (
            sum(travel_times)
            / len(travel_times)
        )

        return round(
            float(average_travel_time),
            self.rounding_precision,
        )

    def get_queue_length(
        self,
        junction_id
    ):
        """
        Calculate junction queue length
        using halted vehicles on controlled
        incoming lanes.
        """

        total_queue = 0

        for lane_id in self.junction_lanes[
            junction_id
        ]:

            total_queue += (
                traci.lane
                .getLastStepHaltingNumber(
                    lane_id
                )
            )

        return int(total_queue)

    def get_traffic_event_type(
        self,
        junction_id
    ):
        """
        Classify junction traffic condition.

        EVENT_NORMAL = 0
        EVENT_SLOW_TRAFFIC = 1
        EVENT_CONGESTION = 2
        """

        vehicle_ids = (
            self.get_junction_vehicle_ids(
                junction_id
            )
        )

        if not vehicle_ids:
            return self.EVENT_NORMAL

        speeds = [
            traci.vehicle.getSpeed(
                vehicle_id
            )
            for vehicle_id in vehicle_ids
        ]

        average_speed = (
            sum(speeds)
            / len(speeds)
        )

        stopped_vehicles = sum(
            speed < self.stopped_speed_threshold
            for speed in speeds
        )

        if (
            stopped_vehicles >= self.congestion_stopped_vehicles
            and average_speed < self.congestion_speed_threshold
        ):
            return self.EVENT_CONGESTION

        if average_speed < self.slow_traffic_speed_threshold:
            return self.EVENT_SLOW_TRAFFIC

        return self.EVENT_NORMAL

    def collect_features(self):
        """
        Collect fixed junction-wise
        traffic feature schema.

        Flow state is calculated once per
        junction per simulation step to avoid
        mutating previous vehicle state more
        than once.
        """

        if not self.initialized:
            self.initialize_junctions()

        junction_features = {}

        for junction_id in self.junction_lanes:

            (
                arrival_rate,
                departure_rate,
            ) = self.get_flow_rates(
                junction_id
            )

            traffic_flow = arrival_rate

            junction_features[
                junction_id
            ] = {
                "vehicle_count": (
                    self.get_vehicle_count(
                        junction_id
                    )
                ),
                "traffic_flow": traffic_flow,
                "arrival_rate": arrival_rate,
                "departure_rate": departure_rate,
                "traffic_event_type": (
                    self.get_traffic_event_type(
                        junction_id
                    )
                ),
                "remaining_green_time": (
                    self.get_remaining_green_time(
                        junction_id
                    )
                ),
                "current_signal_phase": (
                    self.get_current_signal_phase(
                        junction_id
                    )
                ),
                "downstream_occupancy": (
                    self.get_downstream_occupancy(
                        junction_id
                    )
                ),
                "downstream_queue_length": (
                    self.get_downstream_queue_length(
                        junction_id
                    )
                ),
                "average_speed": (
                    self.get_average_speed(
                        junction_id
                    )
                ),
                "waiting_time": (
                    self.get_waiting_time(
                        junction_id
                    )
                ),
                "travel_time": (
                    self.get_travel_time(
                        junction_id
                    )
                ),
                "queue_length": (
                    self.get_queue_length(
                        junction_id
                    )
                ),
            }

        return junction_features