from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector


def main():

    env = SumoEnvironment()
    collector = TrafficCollector()

    env.show_configuration()

    if env.connect():

        print("\nRunning Multi-Junction Simulation...\n")

        event_names = {
            0: "NORMAL",
            1: "SLOW_TRAFFIC",
            2: "CONGESTION",
        }

        try:

            for step in range(50):

                env.simulation_step()

                junction_features = collector.collect_features()

                print(f"\nStep: {step:03d}")

                for junction_id, features in junction_features.items():

                    vehicle_count = features["vehicle_count"]
                    traffic_flow = features["traffic_flow"]
                    event_type = features["traffic_event_type"]
                    remaining_green_time = features["remaining_green_time"]
                    downstream_occupancy = features["downstream_occupancy"]

                    event_name = event_names.get(
                        event_type,
                        "UNKNOWN"
                    )

                    print(
                        f"Junction: {junction_id} | "
                        f"Vehicles: {vehicle_count:02d} | "
                        f"Traffic Flow: {traffic_flow} | "
                        f"Event: {event_name} | "
                        f"Remaining Green Time: {remaining_green_time:.1f} | "
                        f"Downstream Occupancy: {downstream_occupancy:.2f}"
                    )

        finally:

            env.disconnect()


if __name__ == "__main__":
    main()