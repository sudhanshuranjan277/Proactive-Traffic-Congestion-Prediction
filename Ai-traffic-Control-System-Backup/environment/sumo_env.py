"""
SUMO Environment

Responsible for:
- Starting SUMO
- Connecting through TraCI
- Running the simulation
- Restarting SUMO for fresh RL episodes
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
        """
        Display current project and SUMO configuration.
        """

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

    def _validate_configuration(self):
        """
        Validate SUMO configuration before starting simulation.
        """

        if not SUMO_CONFIG:
            raise ValueError(
                "No SUMO configuration selected."
            )

        if not os.path.exists(SUMO_CONFIG):
            raise FileNotFoundError(
                f"SUMO configuration not found: {SUMO_CONFIG}"
            )

    def _build_sumo_command(self):
        """
        Build the SUMO startup command.

        Keeping command creation separate makes the
        environment easier to extend and test.
        """

        return [
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG,
            "--no-step-log",
            "true",
        ]

    def connect(self):
        """
        Start SUMO and establish a TraCI connection.
        """

        if self.connected:
            print("SUMO is already connected.")
            return True

        try:
            self._validate_configuration()

            print("\nStarting SUMO...")

            traci.start(
                self._build_sumo_command()
            )

            self.connected = True

            print(
                "SUMO Connected Successfully."
            )

            return True

        except Exception as error:
            self.connected = False

            print(
                f"Failed to connect with SUMO: "
                f"{error}"
            )

            return False

    def restart(self):
        """
        Restart SUMO from simulation time zero.

        Used by reinforcement learning training so
        every episode starts from a fresh simulation.
        """

        print("\nRestarting SUMO for fresh episode...")

        self.disconnect()

        if not self.connect():
            raise RuntimeError(
                "Failed to restart SUMO."
            )

        simulation_time = self.get_simulation_time()

        if simulation_time != 0.0:
            raise RuntimeError(
                "SUMO restart validation failed. "
                f"Expected simulation time 0.0, "
                f"received {simulation_time}."
            )

        print(
            "SUMO Restarted Successfully "
            f"at simulation time {simulation_time:.0f}s."
        )

        return True

    def simulation_step(self):
        """
        Execute one SUMO simulation step.
        """

        self._require_connection()

        traci.simulationStep()

    def get_simulation_time(self):
        """
        Return current SUMO simulation time.
        """

        self._require_connection()

        return float(
            traci.simulation.getTime()
        )

    def get_vehicle_ids(self):
        """
        Return all active vehicle IDs.
        """

        if not self.connected:
            return []

        return list(
            traci.vehicle.getIDList()
        )

    def get_vehicle_count(self):
        """
        Return total active vehicle count.
        """

        return len(
            self.get_vehicle_ids()
        )

    def _require_connection(self):
        """
        Ensure that SUMO is currently connected.
        """

        if not self.connected:
            raise RuntimeError(
                "SUMO is not connected."
            )

    def disconnect(self):
        """
        Safely close the TraCI connection.
        """

        if not self.connected:
            return

        try:
            traci.close()

        except Exception as error:
            print(
                f"Warning while closing SUMO: "
                f"{error}"
            )

        finally:
            self.connected = False

        print(
            "SUMO Closed Successfully."
        )