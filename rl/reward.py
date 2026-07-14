"""
Reward calculation for traffic signal control.
"""


def compute_reward(
    previous_features,
    current_features,
    action,
    action_executed=True,
):
    if previous_features is None:
        return 0.0

    queue_delta = (
        previous_features["queue_length"]
        - current_features["queue_length"]
    )

    occupancy_delta = (
        previous_features["downstream_occupancy"]
        - current_features["downstream_occupancy"]
    )

    event_penalty = (
        current_features["traffic_event_type"]
        * 0.05
    )

    action_penalty = 0.0

    if action == 1:
        action_penalty = 0.05
    elif action == 2:
        action_penalty = 0.10

    validity_penalty = 0.0

    if not action_executed:
        validity_penalty = 0.20

    reward = (
        queue_delta * 0.8
        + occupancy_delta * 0.2
        - event_penalty
        - action_penalty
        - validity_penalty
    )

    return float(reward)
