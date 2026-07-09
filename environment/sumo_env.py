"""
SUMO Environment

Responsible for:
- Starting SUMO
- Connecting through TraCI
- Running the simulation
- Closing SUMO
"""

import traci
from config import *


class SumoEnvironment:

    def __init__(self):
        self.connected = False

    def show_configuration(self):

        print("=" * 60)
        print(PROJECT_NAME)
        print("=" * 60)

        print(f"Project Root : {PROJECT_ROOT}")
        print(f"SUMO Binary  : {SUMO_BINARY}")
        print(f"Map Folder   : {MAP_DIR}")
        print(f"Data Folder  : {DATA_DIR}")
        print(f"Model Folder : {MODEL_DIR}")
        print(f"Output Folder: {OUTPUT_DIR}")

    def connect(self):

        if SUMO_CONFIG is None:
            print("\nNo SUMO configuration selected.")
            return False

        print("\nStarting SUMO...")

        traci.start([
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG
        ])

        self.connected = True

        print("SUMO Connected Successfully.")

        return True

    def simulation_step(self):
        """
        Execute one simulation step.
        """
        traci.simulationStep()

    def get_vehicle_ids(self):
        """
        Return all vehicle IDs currently in the simulation.
        """
        return traci.vehicle.getIDList()

    def get_vehicle_count(self):
        """
        Return the total number of vehicles currently in the simulation.
        """
        return len(traci.vehicle.getIDList())

    def disconnect(self):

        if self.connected:

            traci.close()

            self.connected = False

            print("SUMO Closed Successfully.")