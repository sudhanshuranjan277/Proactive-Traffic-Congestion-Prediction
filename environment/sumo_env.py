"""
SUMO Environment

Responsible for:
- Starting SUMO
- Connecting through TraCI
- Running the simulation
- Closing SUMO
"""

import os
import traci

from config import (
    PROJECT_NAME,
    PROJECT_ROOT,
    SUMO_BINARY,
    SUMO_CONFIG,
    MAP_DIR,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
)


class SumoEnvironment:

    def __init__(self):
        self.connected = False

    def show_configuration(self):
        """Display current project and SUMO configuration."""

        print("=" * 60)
        print(PROJECT_NAME)
        print("=" * 60)

        print(f"Project Root : {PROJECT_ROOT}")
        print(f"SUMO Binary  : {SUMO_BINARY}")
        print(f"SUMO Config  : {SUMO_CONFIG}")
        print(f"Map Folder   : {MAP_DIR}")
        print(f"Data Folder  : {DATA_DIR}")
        print(f"Model Folder : {MODEL_DIR}")
        print(f"Output Folder: {OUTPUT_DIR}")

    def connect(self):
        """Start SUMO and establish TraCI connection."""

        if not SUMO_CONFIG:
            print("\nNo SUMO configuration selected.")
            return False

        if not os.path.exists(SUMO_CONFIG):
            print(f"\nSUMO configuration not found: {SUMO_CONFIG}")
            return False

        print("\nStarting SUMO...")

        try:
            traci.start([
                SUMO_BINARY,
                "-c",
                SUMO_CONFIG,
                "--no-step-log",
                "true",
            ])

            self.connected = True

            print("SUMO Connected Successfully.")

            return True

        except Exception as error:
            print(f"Failed to connect with SUMO: {error}")
            return False

    def simulation_step(self):
        """Execute one SUMO simulation step."""

        if not self.connected:
            raise RuntimeError("SUMO is not connected.")

        traci.simulationStep()

    def get_vehicle_ids(self):
        """Return all active vehicle IDs."""

        if not self.connected:
            return []

        return traci.vehicle.getIDList()

    def get_vehicle_count(self):
        """Return total active vehicle count."""

        return len(self.get_vehicle_ids())

    def disconnect(self):
        """Safely close the TraCI connection."""

        if self.connected:

            traci.close()

            self.connected = False

            print("SUMO Closed Successfully.")