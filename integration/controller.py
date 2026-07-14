"""
Traffic signal controller for safe SUMO action execution.
"""

import traci


class TrafficSignalAction:
    MAINTAIN = 0
    EXTEND_GREEN = 1
    NEXT_PHASE = 2

    NAMES = {
        MAINTAIN: "MAINTAIN",
        EXTEND_GREEN: "EXTEND_GREEN",
        NEXT_PHASE: "NEXT_PHASE",
    }


class TrafficSignalController:

    def __init__(self, extension_seconds=5):
        self.extension_seconds = extension_seconds

    def get_current_phase(self, junction_id):
        return traci.trafficlight.getPhase(
            junction_id
        )

    def get_phase_number(self, junction_id):
        return traci.trafficlight.getPhaseNumber(
            junction_id
        )

    def get_current_phase_state(self, junction_id):
        return traci.trafficlight.getRedYellowGreenState(
            junction_id
        )

    def is_current_phase_green(self, junction_id):
        phase_state = self.get_current_phase_state(
            junction_id
        )
        return "G" in phase_state or "g" in phase_state

    def apply_action(
        self,
        junction_id,
        action,
    ):
        if action == TrafficSignalAction.MAINTAIN:
            return True

        if (
            action == TrafficSignalAction.EXTEND_GREEN
            and self.is_current_phase_green(junction_id)
        ):
            current_duration = (
                traci.trafficlight.getPhaseDuration(
                    junction_id
                )
            )

            new_duration = (
                current_duration
                + self.extension_seconds
            )

            traci.trafficlight.setPhaseDuration(
                junction_id,
                new_duration,
            )

            return True

        if action == TrafficSignalAction.NEXT_PHASE:
            current_phase = self.get_current_phase(
                junction_id
            )

            phase_count = self.get_phase_number(
                junction_id
            )

            next_phase = (
                current_phase + 1
            ) % phase_count

            traci.trafficlight.setPhase(
                junction_id,
                next_phase,
            )

            return True

        return False
