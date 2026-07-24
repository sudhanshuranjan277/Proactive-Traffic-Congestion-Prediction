# Proactive Traffic Congestion Prediction and Signal Control

## 1. System Objective

The system predicts future traffic congestion before it occurs and proactively controls traffic signals using Deep Reinforcement Learning.

The system integrates:

- OpenStreetMap (OSM)
- SUMO Traffic Simulator
- TraCI
- Traffic Feature Collection
- LSTM-Based Traffic Prediction
- Double Deep Q-Network (DDQN)
- Adaptive Traffic Signal Control


## 2. System Architecture

Multiple OSM Locations
        |
        v
SUMO Traffic Simulation
        |
        v
TraCI Real-Time Interface
        |
        v
Dynamic Multi-Junction Feature Collector
        |
        v
Traffic Feature Validation
        |
        v
Location-Wise Traffic Dataset
        |
        v
Combined Traffic Dataset
        |
        v
LSTM Traffic Prediction
        |
        v
Future Queue Length and Saturation Prediction
        |
        v
DDQN Reinforcement Learning Agent
        |
        v
Traffic Signal Decision
        |
        v
TraCI Signal Controller
        |
        v
SUMO Simulation


## 3. Multi-Location Traffic Simulation

The system supports multiple OSM-based SUMO traffic networks.

Each location has an independent SUMO configuration.

Example:

maps/
    location_1/
        osm.net.xml.gz
        osm.sumocfg

    location_2/
        osm.net.xml.gz
        osm.sumocfg

    location_3/
        osm.net.xml.gz
        osm.sumocfg

Traffic data is collected independently for every location.


## 4. Dynamic Junction Discovery

Traffic signal junctions are automatically discovered using TraCI.

For every traffic light:

1. Get Traffic Light ID.
2. Identify controlled incoming lanes.
3. Identify downstream lanes.
4. Track vehicles on incoming lanes.
5. Read traffic signal state.

No traffic junction IDs or detector IDs should be permanently hardcoded.


## 5. Traffic Features

### 5.1 Traffic Flow

Traffic flow represents vehicles entering the controlled junction lanes during a defined time window.

Recommended unit:

vehicles per minute

Traffic flow is calculated using newly observed vehicle IDs on incoming controlled lanes.


### 5.2 Traffic Event Type

Traffic conditions are classified using average vehicle speed and stopped vehicles.

Event classes:

0 = NORMAL

1 = SLOW_TRAFFIC

2 = CONGESTION

The event must be calculated independently for every junction.


### 5.3 Remaining Green Time

Remaining green time represents the remaining duration of the active green signal phase.

Calculation:

Remaining Green Time = Next Signal Switch Time - Current Simulation Time

If no green signal is active, the value is zero.


### 5.4 Downstream Occupancy

Downstream occupancy measures traffic density on outgoing lanes connected to the junction.

The average occupancy of all downstream lanes is calculated.

This feature helps identify traffic spillback conditions.


### 5.5 Queue Length

Queue length represents the number of halted vehicles on incoming controlled lanes.

Vehicles with a very low speed or SUMO halting status are considered queued vehicles.

Queue length is calculated independently for every junction.


## 6. Feature Collection Window

SUMO simulation step:

1 second

Traffic observation window:

60 simulation seconds

A dataset observation is generated every 60 simulation seconds.

Example:

Time 60 seconds -> Dataset Row 1

Time 120 seconds -> Dataset Row 2

Time 180 seconds -> Dataset Row 3


## 7. Feature Schema

Every dataset observation follows a fixed feature schema.

Fields:

location_id

junction_id

simulation_time

vehicle_count

traffic_flow

traffic_event_type

remaining_green_time

downstream_occupancy

queue_length


## 8. Dataset Architecture

Separate datasets are generated for every OSM location.

Example:

data/processed/location_1_dataset.csv

data/processed/location_2_dataset.csv

data/processed/location_3_dataset.csv

The location datasets are merged to generate:

data/processed/combined_traffic_dataset.csv


## 9. LSTM Traffic Prediction

The LSTM model analyzes historical junction-specific traffic patterns.

Input features:

Traffic Flow

Traffic Event Type

Remaining Green Time

Downstream Occupancy

Queue Length

Historical observation window:

30 observations

Prediction horizon:

10 observations

The LSTM prediction target is:

Future Queue Length

Input tensor shape:

(batch_size, 30, 5)

Output tensor shape:

(batch_size, 10, 1)

The model predicts ten future queue length observations.

Traffic saturation is not directly predicted by the LSTM.

Saturation is calculated as a derived traffic control metric using the predicted queue length and the physical queue capacity of the active SUMO junction.

### 10. Traffic Saturation

Traffic saturation represents predicted queue pressure relative to the physical queue capacity of a signalized junction.

The future queue sequence predicted by the LSTM is:

Q = [q1, q2, q3, ..., q10]

The following queue indicators are derived:

Peak Predicted Queue:

Q_peak = max(Q)

Mean Predicted Queue:

Q_mean = mean(Q)

Final Predicted Queue:

Q_final = q10

Queue Growth:

Q_growth = Q_final - q1


### Dynamic Junction Queue Capacity

Junction queue capacity must be derived from the active SUMO network.

For every controlled incoming lane:

Lane Capacity = floor(
    Lane Length / Vehicle Space
)

Vehicle Space represents the estimated physical road space occupied by one queued vehicle including its minimum gap.

Junction Capacity is calculated as:

Junction Capacity = Sum of Incoming Lane Capacities

No fixed junction capacity value should be manually hardcoded.


### Saturation Calculation

Peak Saturation:

Peak Saturation = Q_peak / Junction Capacity

Mean Saturation:

Mean Saturation = Q_mean / Junction Capacity

Saturation values are bounded between 0 and 1.

If junction capacity is invalid or unavailable, the condition must be handled explicitly.

The system must not silently substitute a manually predetermined capacity value.


## 11. DDQN Reinforcement Learning

The Double Deep Q-Network agent receives the predicted future traffic state.

State may include:

Predicted Queue Length

Predicted Saturation

Current Traffic Flow

Remaining Green Time

Downstream Occupancy

Possible actions:

Maintain Current Signal Phase

Extend Green Time

Switch Signal Phase

The final action space must respect the traffic signal program and safe phase transitions.


## 12. Reward Function

The reinforcement learning reward function should minimize:

Queue Length

Vehicle Waiting Time

Traffic Congestion

Downstream Spillback

The reward function may include penalties for unsafe or excessive signal switching.


## 13. Traffic Signal Control

The DDQN agent selects a traffic signal action.

The signal controller applies the action through TraCI.

The SUMO simulation advances to the next step.

The process repeats continuously.


## 14. Final System Flow

Traffic Observation

        |

Feature Collection

        |

Feature Validation

        |

Traffic Dataset

        |

LSTM Prediction

        |

Future Queue and Saturation

        |

DDQN Decision

        |

Signal Control

        |

SUMO Simulation

        |

Next Traffic Observation


## 15. Implementation Rule

The system must follow a fixed feature interface.

Feature names and data types must remain consistent between:

collector.py

pipeline.py

dataset generation

LSTM preprocessing

LSTM prediction

DDQN state generation

evaluation

Feature values must originate from the active SUMO simulation and the loaded OSM network.

Output values must not be manually predetermined.