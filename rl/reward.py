"""
Reward calculation for proactive
traffic signal control.

The reward is calculated from real
consecutive SUMO traffic observations.
"""

from integration.collector import TrafficCollector
from integration.controller import TrafficSignalAction
from config import (
    REWARD_QUEUE_WEIGHT,
    REWARD_WAITING_TIME_WEIGHT,
    REWARD_OCCUPANCY_WEIGHT,
    REWARD_DOWNSTREAM_QUEUE_WEIGHT,
    REWARD_SPEED_WEIGHT,
    REWARD_SLOW_TRAFFIC_PENALTY,
    REWARD_CONGESTION_PENALTY,
    REWARD_EXTEND_PENALTY,
    REWARD_NEXT_PHASE_PENALTY,
    REWARD_INVALID_ACTION_PENALTY,
)


def compute_reward(
    previous_features,
    current_features,
    action,
    action_executed=True,
):
    """
    Calculate DDQN reward from changes
    in real SUMO traffic conditions.
    """

    if previous_features is None:
        return 0.0

    previous_queue = float(
        previous_features[
            "queue_length"
        ]
    )

    current_queue = float(
        current_features[
            "queue_length"
        ]
    )

    previous_waiting_time = float(
        previous_features[
            "waiting_time"
        ]
    )

    current_waiting_time = float(
        current_features[
            "waiting_time"
        ]
    )

    previous_occupancy = float(
        previous_features[
            "downstream_occupancy"
        ]
    )

    current_occupancy = float(
        current_features[
            "downstream_occupancy"
        ]
    )

    previous_downstream_queue = float(
        previous_features[
            "downstream_queue_length"
        ]
    )

    current_downstream_queue = float(
        current_features[
            "downstream_queue_length"
        ]
    )

    previous_speed = float(
        previous_features[
            "average_speed"
        ]
    )

    current_speed = float(
        current_features[
            "average_speed"
        ]
    )

    traffic_event_type = int(
        current_features[
            "traffic_event_type"
        ]
    )

    queue_improvement = (
        previous_queue
        - current_queue
    )

    waiting_time_improvement = (
        previous_waiting_time
        - current_waiting_time
    )

    occupancy_improvement = (
        previous_occupancy
        - current_occupancy
    )

    downstream_queue_improvement = (
        previous_downstream_queue
        - current_downstream_queue
    )

    speed_improvement = (
        current_speed
        - previous_speed
    )

    reward = 0.0

    reward += (
        REWARD_QUEUE_WEIGHT
        * queue_improvement
    )

    reward += (
        REWARD_WAITING_TIME_WEIGHT
        * waiting_time_improvement
    )

    reward += (
        REWARD_OCCUPANCY_WEIGHT
        * occupancy_improvement
    )

    reward += (
        REWARD_DOWNSTREAM_QUEUE_WEIGHT
        * downstream_queue_improvement
    )

    reward += (
        REWARD_SPEED_WEIGHT
        * speed_improvement
    )

    if traffic_event_type == TrafficCollector.EVENT_SLOW_TRAFFIC:

        reward -= (
            REWARD_SLOW_TRAFFIC_PENALTY
        )

    elif traffic_event_type == TrafficCollector.EVENT_CONGESTION:

        reward -= (
            REWARD_CONGESTION_PENALTY
        )

    if action == TrafficSignalAction.EXTEND_CURRENT_PHASE:

        reward -= (
            REWARD_EXTEND_PENALTY
        )

    elif action == TrafficSignalAction.MOVE_TO_NEXT_PHASE:

        reward -= (
            REWARD_NEXT_PHASE_PENALTY
        )

    if not action_executed:

        reward -= (
            REWARD_INVALID_ACTION_PENALTY
        )

    return float(
        reward
    )