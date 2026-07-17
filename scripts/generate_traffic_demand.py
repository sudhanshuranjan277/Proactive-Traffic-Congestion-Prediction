"""
Junction-Aware Scenario Traffic Demand Generator

Generates temporally varying passenger traffic
through dynamically discovered signalized junctions.

Traffic pattern:

LOW
    ->
MEDIUM
    ->
HIGH
    ->
PEAK
    ->
RECOVERY

Signalized junctions and controlled traffic
connections are discovered directly from the
active SUMO OSM network.

No junction IDs, edge IDs, traffic features,
or model outputs are manually predetermined.
"""

import os
import sys
import random
import xml.etree.ElementTree as ET


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


SUMO_HOME = os.environ.get(
    "SUMO_HOME"
)

if not SUMO_HOME:
    raise EnvironmentError(
        "SUMO_HOME environment variable "
        "is not configured."
    )


SUMO_TOOLS = os.path.join(
    SUMO_HOME,
    "tools",
)

if SUMO_TOOLS not in sys.path:
    sys.path.insert(
        0,
        SUMO_TOOLS,
    )


import sumolib


OSM_FOLDER = os.path.join(
    PROJECT_ROOT,
    "maps",
    "osm",
)

NETWORK_FILE = os.path.join(
    OSM_FOLDER,
    "osm.net.xml.gz",
)

OUTPUT_TRIP_FILE = os.path.join(
    OSM_FOLDER,
    "scenario_passenger.trips.xml",
)


SIMULATION_END = 3600

RANDOM_SEED = 42


TRAFFIC_SCENARIOS = [
    {
        "name": "low",
        "begin": 0,
        "end": 900,
        "insertion_rate": 300,
    },
    {
        "name": "medium",
        "begin": 900,
        "end": 1800,
        "insertion_rate": 600,
    },
    {
        "name": "high",
        "begin": 1800,
        "end": 2700,
        "insertion_rate": 900,
    },
    {
        "name": "peak",
        "begin": 2700,
        "end": 3300,
        "insertion_rate": 1200,
    },
    {
        "name": "recovery",
        "begin": 3300,
        "end": SIMULATION_END,
        "insertion_rate": 300,
    },
]


def validate_network_file():
    """
    Validate SUMO network availability.
    """

    if not os.path.isfile(
        NETWORK_FILE
    ):
        raise FileNotFoundError(
            f"SUMO network file not found: "
            f"{NETWORK_FILE}"
        )


def load_network():
    """
    Load active SUMO OSM network.
    """

    print(
        "\nLoading SUMO network..."
    )

    network = sumolib.net.readNet(
        NETWORK_FILE,
        withPrograms=True,
    )

    return network


def is_passenger_allowed(
    edge
):
    """
    Return True when at least one edge lane
    permits passenger vehicles.
    """

    for lane in edge.getLanes():

        if lane.allows(
            "passenger"
        ):
            return True

    return False


def discover_signal_connections(
    network
):
    """
    Dynamically discover valid controlled
    passenger-vehicle connections through
    every signalized junction.

    Each connection contains:

    traffic light ID
    incoming edge
    outgoing edge
    """

    signal_connections = []

    traffic_lights = (
        network.getTrafficLights()
    )

    for traffic_light in traffic_lights:

        junction_id = (
            traffic_light.getID()
        )

        discovered_pairs = set()

        for connection in (
            traffic_light.getConnections()
        ):

            incoming_lane = connection[0]
            outgoing_lane = connection[1]

            incoming_edge = (
                incoming_lane.getEdge()
            )

            outgoing_edge = (
                outgoing_lane.getEdge()
            )

            incoming_edge_id = (
                incoming_edge.getID()
            )

            outgoing_edge_id = (
                outgoing_edge.getID()
            )

            if incoming_edge_id.startswith(
                ":"
            ):
                continue

            if outgoing_edge_id.startswith(
                ":"
            ):
                continue

            if not is_passenger_allowed(
                incoming_edge
            ):
                continue

            if not is_passenger_allowed(
                outgoing_edge
            ):
                continue

            connection_pair = (
                incoming_edge_id,
                outgoing_edge_id,
            )

            if connection_pair in (
                discovered_pairs
            ):
                continue

            discovered_pairs.add(
                connection_pair
            )

            signal_connections.append({
                "junction_id": junction_id,
                "from_edge": incoming_edge_id,
                "to_edge": outgoing_edge_id,
            })

    if not signal_connections:

        raise ValueError(
            "No passenger-compatible "
            "signalized traffic connections "
            "were discovered."
        )

    return signal_connections


def calculate_trip_count(
    scenario
):
    """
    Calculate scenario trip count from
    insertion rate and scenario duration.
    """

    duration_seconds = (
        scenario["end"]
        - scenario["begin"]
    )

    trip_count = round(
        scenario["insertion_rate"]
        * duration_seconds
        / 3600
    )

    return int(
        trip_count
    )


def create_departure_times(
    scenario,
    trip_count,
):
    """
    Generate evenly distributed departure
    times with small random temporal jitter.
    """

    if trip_count <= 0:
        return []

    begin = scenario["begin"]
    end = scenario["end"]

    duration = end - begin

    interval = (
        duration
        / trip_count
    )

    departure_times = []

    for trip_index in range(
        trip_count
    ):

        base_departure = (
            begin
            + (
                trip_index
                + 0.5
            )
            * interval
        )

        jitter_limit = min(
            interval * 0.35,
            5.0,
        )

        jitter = random.uniform(
            -jitter_limit,
            jitter_limit,
        )

        departure_time = (
            base_departure
            + jitter
        )

        departure_time = max(
            float(begin),
            departure_time,
        )

        departure_time = min(
            float(end) - 0.01,
            departure_time,
        )

        departure_times.append(
            round(
                departure_time,
                2,
            )
        )

    departure_times.sort()

    return departure_times


def generate_scenario_trips(
    scenario,
    signal_connections,
):
    """
    Generate trips through dynamically
    discovered signalized connections.
    """

    trip_count = calculate_trip_count(
        scenario
    )

    departure_times = (
        create_departure_times(
            scenario,
            trip_count,
        )
    )

    scenario_trips = []

    shuffled_connections = (
        signal_connections.copy()
    )

    random.shuffle(
        shuffled_connections
    )

    for trip_index, departure_time in enumerate(
        departure_times
    ):

        connection = shuffled_connections[
            trip_index
            % len(shuffled_connections)
        ]

        trip = {
            "id": (
                f"{scenario['name']}_"
                f"{trip_index:05d}"
            ),
            "depart": (
                f"{departure_time:.2f}"
            ),
            "from": connection[
                "from_edge"
            ],
            "to": connection[
                "to_edge"
            ],
            "departLane": "best",
            "departSpeed": "max",
            "junction_id": connection[
                "junction_id"
            ],
            "scenario": scenario[
                "name"
            ],
        }

        scenario_trips.append(
            trip
        )

    return scenario_trips


def generate_all_trips(
    signal_connections
):
    """
    Generate complete multi-scenario
    traffic demand.
    """

    all_trips = []

    for scenario in TRAFFIC_SCENARIOS:

        print(
            f"\nGenerating "
            f"{scenario['name'].upper()} "
            f"traffic..."
        )

        print(
            f"Time Window    : "
            f"{scenario['begin']}s "
            f"- {scenario['end']}s"
        )

        print(
            f"Insertion Rate : "
            f"{scenario['insertion_rate']} "
            f"vehicles/hour"
        )

        scenario_trips = (
            generate_scenario_trips(
                scenario,
                signal_connections,
            )
        )

        print(
            f"Generated Trips: "
            f"{len(scenario_trips)}"
        )

        all_trips.extend(
            scenario_trips
        )

    all_trips.sort(
        key=lambda trip: float(
            trip["depart"]
        )
    )

    return all_trips


def save_trip_file(
    trips
):
    """
    Save generated SUMO passenger trips.
    """

    routes = ET.Element(
        "routes"
    )

    for trip_data in trips:

        trip_attributes = {
            "id": trip_data["id"],
            "depart": trip_data["depart"],
            "from": trip_data["from"],
            "to": trip_data["to"],
            "departLane": trip_data[
                "departLane"
            ],
            "departSpeed": trip_data[
                "departSpeed"
            ],
        }

        ET.SubElement(
            routes,
            "trip",
            trip_attributes,
        )

    tree = ET.ElementTree(
        routes
    )

    ET.indent(
        tree,
        space="    ",
    )

    tree.write(
        OUTPUT_TRIP_FILE,
        encoding="utf-8",
        xml_declaration=True,
    )


def validate_generated_demand(
    trips
):
    """
    Validate scenario and junction
    demand distribution.
    """

    if not trips:

        raise ValueError(
            "No passenger trips generated."
        )

    departure_times = [
        float(
            trip["depart"]
        )
        for trip in trips
    ]

    print(
        "\nGenerated Demand Validation"
    )

    print(
        "=" * 60
    )

    print(
        f"Total Trips  : "
        f"{len(trips)}"
    )

    print(
        f"First Depart : "
        f"{min(departure_times):.2f}s"
    )

    print(
        f"Last Depart  : "
        f"{max(departure_times):.2f}s"
    )

    print(
        "\nScenario Distribution:"
    )

    for scenario in TRAFFIC_SCENARIOS:

        scenario_count = sum(
            trip["scenario"]
            == scenario["name"]
            for trip in trips
        )

        print(
            f"{scenario['name'].upper():10s} | "
            f"{scenario['begin']:4d}s - "
            f"{scenario['end']:4d}s | "
            f"Trips: {scenario_count}"
        )

    print(
        "\nJunction Demand Distribution:"
    )

    junction_ids = sorted(
        {
            trip["junction_id"]
            for trip in trips
        }
    )

    for junction_id in junction_ids:

        junction_count = sum(
            trip["junction_id"]
            == junction_id
            for trip in trips
        )

        print(
            f"{junction_id} | "
            f"Trips: {junction_count}"
        )


def main():

    print(
        "=" * 60
    )

    print(
        "Junction-Aware SUMO Traffic "
        "Demand Generator"
    )

    print(
        "=" * 60
    )

    random.seed(
        RANDOM_SEED
    )

    validate_network_file()

    network = load_network()

    signal_connections = (
        discover_signal_connections(
            network
        )
    )

    print(
        f"Signalized Junctions : "
        f"{len(network.getTrafficLights())}"
    )

    print(
        f"Controlled Passenger "
        f"Connections : "
        f"{len(signal_connections)}"
    )

    trips = generate_all_trips(
        signal_connections
    )

    save_trip_file(
        trips
    )

    validate_generated_demand(
        trips
    )

    print(
        "\nTraffic Demand Generation "
        "Completed."
    )

    print(
        f"Trip File : "
        f"{OUTPUT_TRIP_FILE}"
    )


if __name__ == "__main__":
    main()