from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector


def main():

    # Initialize Environment
    env = SumoEnvironment()

    # Initialize Traffic Collector
    collector = TrafficCollector()

    # Display Project Configuration
    env.show_configuration()

    # Connect to SUMO
    if env.connect():

        print("\nRunning Simulation...\n")

        for step in range(50):

            # Move simulation by one step
            env.simulation_step()

            # Collect Traffic Features
            vehicle_count = collector.get_vehicle_count()
            traffic_flow = collector.get_traffic_flow()
            features = collector.collect_features()

            # Display Live Data
            print(
                f"Step: {step:03d} | "
                f"Vehicles: {vehicle_count:02d} | "
                f"Traffic Flow: {traffic_flow}"
            )

        # Close SUMO
        env.disconnect()


if __name__ == "__main__":
    main()