"""
Traffic Signal Controller

Responsible for safely executing reinforcement
learning traffic signal actions in SUMO.
"""

from enum import IntEnum

import traci


class TrafficSignalAction(IntEnum):
    """
    Available DDQN traffic signal actions.
    """

    KEEP_CURRENT_PHASE = 0

    EXTEND_CURRENT_PHASE = 1

    MOVE_TO_NEXT_PHASE = 2


class TrafficSignalController:

    def __init__(
        self,
        extension_seconds=5,
    ):
        if extension_seconds <= 0:
            raise ValueError(
                "extension_seconds must be "
                "greater than zero."
            )

        self.extension_seconds = float(
            extension_seconds
        )

    def get_current_phase(
        self,
        junction_id,
    ):
        """
        Return the current traffic signal phase index.
        """

        return int(
            traci.trafficlight.getPhase(
                junction_id
            )
        )

    def get_phase_number(
        self,
        junction_id,
    ):
        """
        Return the number of phases in the currently
        active SUMO traffic light program.
        """

        current_program = (
            traci.trafficlight.getProgram(
                junction_id
            )
        )

        program_logics = (
            traci.trafficlight
            .getAllProgramLogics(
                junction_id
            )
        )

        if not program_logics:
            raise RuntimeError(
                "No traffic light program logic "
                "found for junction: "
                f"{junction_id}"
            )

        active_logic = None

        for logic in program_logics:

            if (
                str(logic.programID)
                == str(current_program)
            ):
                active_logic = logic
                break

        if active_logic is None:
            raise RuntimeError(
                "Active traffic light program "
                "could not be resolved for junction "
                f"{junction_id}. "
                f"Current program: {current_program}"
            )

        phase_count = len(
            active_logic.phases
        )

        if phase_count <= 0:
            raise RuntimeError(
                "Traffic light program contains "
                "no phases for junction: "
                f"{junction_id}"
            )

        return int(
            phase_count
        )

    def get_remaining_phase_time(
        self,
        junction_id,
    ):
        """
        Return remaining time before the next
        scheduled traffic signal switch.
        """

        simulation_time = float(
            traci.simulation.getTime()
        )

        next_switch = float(
            traci.trafficlight.getNextSwitch(
                junction_id
            )
        )

        return max(
            0.0,
            next_switch - simulation_time,
        )

    def keep_current_phase(
        self,
        junction_id,
    ):
        """
        Keep the current SUMO signal behaviour.

        No manual phase modification is required.
        """

        return True

    def extend_current_phase(
        self,
        junction_id,
    ):
        """
        Extend the remaining duration of the
        current traffic signal phase.
        """

        remaining_time = (
            self.get_remaining_phase_time(
                junction_id
            )
        )

        new_duration = (
            remaining_time
            + self.extension_seconds
        )

        traci.trafficlight.setPhaseDuration(
            junction_id,
            new_duration,
        )

        return True

    def move_to_next_phase(
        self,
        junction_id,
    ):
        """
        Move to the next phase in the currently
        active SUMO traffic light program.
        """

        current_phase = (
            self.get_current_phase(
                junction_id
            )
        )

        phase_count = (
            self.get_phase_number(
                junction_id
            )
        )

        next_phase = (
            current_phase + 1
        ) % phase_count

        traci.trafficlight.setPhase(
            junction_id,
            next_phase,
        )

        return True

    def apply_action(
        self,
        junction_id,
        action,
    ):
        """
        Execute one DDQN traffic signal action.

        Returns True when the requested action is
        successfully executed and False when the
        action is invalid or SUMO rejects it.
        """

        try:

            action = TrafficSignalAction(
                int(action)
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        try:

            if (
                action
                == TrafficSignalAction
                .KEEP_CURRENT_PHASE
            ):

                return (
                    self.keep_current_phase(
                        junction_id
                    )
                )

            if (
                action
                == TrafficSignalAction
                .EXTEND_CURRENT_PHASE
            ):

                return (
                    self.extend_current_phase(
                        junction_id
                    )
                )

            if (
                action
                == TrafficSignalAction
                .MOVE_TO_NEXT_PHASE
            ):

                return (
                    self.move_to_next_phase(
                        junction_id
                    )
                )

        except traci.TraCIException:

            return False

        return False