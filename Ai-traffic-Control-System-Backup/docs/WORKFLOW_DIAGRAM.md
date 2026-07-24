# AI Traffic Control System - Complete Workflow Diagram

---

## 📊 COMPLETE END-TO-END WORKFLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AI TRAFFIC CONTROL SYSTEM WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘


═════════════════════════════════════════════════════════════════════════════════
                              PHASE 1: DATA COLLECTION
═════════════════════════════════════════════════════════════════════════════════

                                    SUMO Simulator
                                    ┌──────────┐
                                    │  SUMO    │
                                    │ Network  │
                                    └─────┬────┘
                                          │
                        [Step simulation every 1 second]
                                          │
                                          ▼
                            ┌──────────────────────────┐
                            │  Traffic Collector       │
                            │  (TrafficCollector)      │
                            └──────────┬───────────────┘
                                       │
            [Collects 13 metrics per junction per timestep]
            
            📊 Collected Features (per junction, per second):
            ├─ vehicle_count: Total vehicles at junction
            ├─ traffic_flow: New arrivals
            ├─ arrival_rate: Vehicles entering
            ├─ departure_rate: Vehicles leaving
            ├─ traffic_event_type: (Normal/Slow/Congestion)
            ├─ remaining_green_time: Active green duration
            ├─ current_signal_phase: Phase index
            ├─ downstream_occupancy: % lanes occupied downstream
            ├─ downstream_queue_length: Halted vehicles ahead
            ├─ average_speed: Mean vehicle velocity
            ├─ waiting_time: Average wait duration
            ├─ travel_time: Trip time for vehicles
            └─ queue_length: Halted vehicles (TARGET)
                                       │
                    [Data flows to Traffic Pipeline]


═════════════════════════════════════════════════════════════════════════════════
                     PHASE 2: DATA AGGREGATION & STORAGE
═════════════════════════════════════════════════════════════════════════════════

                            Traffic Pipeline
                           ┌──────────────────┐
                           │ (TrafficPipeline)│
                           └────────┬─────────┘
                                    │
        [Collects observations in sliding window]
        [Window size: observation_window = 60 seconds]
                                    │
        Sliding Window [60 timesteps × 13 features]
                                    │
                    ┌───────────────┴──────────────┐
                    │                              │
           Aggregation Logic:                    Output Format:
           ├─ Latest observation row          ├─ CSV Record
           ├─ Historical buffer               ├─ Path: data/processed/
           ├─ Duplicate detection             ├─ File: location_1_dataset.csv
           └─ Consistency validation          └─ Columns: All 13 features
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Dataset CSV File         │
                    │  (location_1_dataset.csv) │
                    │                           │
                    │  Rows: ~3600 (1 hour)    │
                    │  Cols: 13 features       │
                    └───────────┬───────────────┘
                                │


═════════════════════════════════════════════════════════════════════════════════
                      PHASE 3: DATA PREPROCESSING
═════════════════════════════════════════════════════════════════════════════════

    Raw CSV → Preprocessing Pipeline
              ┌───────────────────────────┐
              │    1. Data Cleaning       │
              │    ├─ Remove duplicates   │
              │    ├─ Replace ±∞ values  │
              │    ├─ Fill missing values │
              │    └─ Handle outliers    │
              └────────────┬──────────────┘
                           │
              ┌────────────▼───────────────┐
              │  2. Feature Normalization  │
              │                            │
              │  MinMaxScaler([0, 1])     │
              │  ├─ Feature scaling        │
              │  ├─ Target scaling         │
              │  └─ Save scalers.pkl      │
              └────────────┬───────────────┘
                           │
              ┌────────────▼──────────────────┐
              │  3. Sequence Generation       │
              │                               │
              │  Sliding window approach:    │
              │  ├─ Lookback: 30 timesteps   │
              │  ├─ Horizon: 10 timesteps    │
              │  ├─ Input:  30 × 12 features │
              │  └─ Target: 10 × 1 value     │
              └────────────┬──────────────────┘
                           │
              ┌────────────▼──────────────────┐
              │  4. Train-Test Split         │
              │  ├─ Train: 80%               │
              │  ├─ Test: 20%                │
              │  └─ Shuffle: False (temporal)│
              └────────────┬──────────────────┘
                           │
              ┌────────────▼──────────────────┐
              │  5. DataLoader Creation      │
              │  ├─ Batch size: 32           │
              │  ├─ Shuffle: True (training) │
              │  └─ Num workers: 0           │
              └────────────┬──────────────────┘
                           │
                    Ready for Training ✓


═════════════════════════════════════════════════════════════════════════════════
                      PHASE 4: LSTM MODEL TRAINING
═════════════════════════════════════════════════════════════════════════════════

    Training Loop (100 epochs max, early stop at patience=10)

    ┌─────────────────────────────────────────┐
    │  LSTM Model Architecture                │
    │                                         │
    │  Input: [Batch×30×12]                  │
    │     ↓                                   │
    │  LSTM Layer 1: 12→64                   │
    │     ↓                                   │
    │  Dropout: 0.2                          │
    │     ↓                                   │
    │  LSTM Layer 2: 64→64                   │
    │     ↓                                   │
    │  Final Hidden: [Batch×64]              │
    │     ↓                                   │
    │  Output Linear: 64→10                  │
    │     ↓                                   │
    │  Reshape: [Batch×10×1]                 │
    └─────────┬───────────────────────────────┘
              │
              │ For each epoch:
              │
    ┌─────────▼──────────────────────────────┐
    │  1. TRAINING LOOP                       │
    │     For each batch:                     │
    │     ├─ Forward pass → predictions       │
    │     ├─ Compute MSELoss                  │
    │     ├─ Backward pass                    │
    │     ├─ Clip gradients (max_norm=1.0)   │
    │     └─ Optimizer step (Adam)            │
    └─────────┬──────────────────────────────┘
              │
    ┌─────────▼──────────────────────────────┐
    │  2. VALIDATION LOOP                     │
    │     For each batch (no gradient):      │
    │     ├─ Forward pass                     │
    │     ├─ Compute loss                     │
    │     └─ Record metrics                   │
    └─────────┬──────────────────────────────┘
              │
    ┌─────────▼──────────────────────────────┐
    │  3. LEARNING RATE SCHEDULING            │
    │     If val_loss not improving:          │
    │     ├─ LR *= 0.5                        │
    │     ├─ Min LR: 1e-6                     │
    │     └─ Patience: 5 epochs               │
    └─────────┬──────────────────────────────┘
              │
    ┌─────────▼──────────────────────────────┐
    │  4. EARLY STOPPING CHECK                │
    │     If val_loss not improving:          │
    │     ├─ Patience counter++               │
    │     └─ Stop if patience=10              │
    └─────────┬──────────────────────────────┘
              │
    ┌─────────▼──────────────────────────────┐
    │  5. BEST MODEL SAVING                   │
    │     If val_loss < best_loss:            │
    │     ├─ Save model weights               │
    │     ├─ Reset patience counter           │
    │     └─ Update best_loss                 │
    └─────────┬──────────────────────────────┘
              │
        [Repeat for next epoch]
              │
    Expected Output:
    ├─ models/lstm_model.pth (weights)
    ├─ models/lstm_scalers.pkl (normalization)
    ├─ outputs/training_history.csv (epoch stats)
    ├─ outputs/metrics.csv (final metrics)
    ├─ outputs/loss_curve.png (training plots)
    └─ outputs/prediction_vs_actual.png (validation plots)


═════════════════════════════════════════════════════════════════════════════════
                    PHASE 5: LSTM MODEL EVALUATION
═════════════════════════════════════════════════════════════════════════════════

    Validation Metrics Computation
    ┌─────────────────────────────┐
    │  Test Set Predictions       │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Metric Calculation         │
    │                             │
    │  MAE = Σ|y_true - y_pred|/n│
    │  RMSE = √(Σ(y_true-y_pred)²/n)
    │  R² = 1 - Σ(y_true-y_pred)²│
    │           ─────────────────  │
    │           Σ(y_true-ȳ)²      │
    └──────────────┬──────────────┘
                   │
    Sample Results:
    ├─ MAE: 1.2-2.5 vehicles
    ├─ RMSE: 2.0-4.0 vehicles
    ├─ R²: 0.85-0.95
    └─ Training Time: 10-20 minutes


═════════════════════════════════════════════════════════════════════════════════
                      PHASE 6: REAL-TIME PREDICTION
═════════════════════════════════════════════════════════════════════════════════

    During DDQN Training / Real-time Control

    ┌──────────────────────────────┐
    │  1. OBSERVATION COLLECTION   │
    │     Collect last 30 steps    │
    │     Shape: [30 × 12]         │
    └────────────┬─────────────────┘
                 │
    ┌────────────▼─────────────────┐
    │  2. PREPROCESSING            │
    │     Apply feature_scaler      │
    │     Shape: [30 × 12] → [0,1] │
    └────────────┬─────────────────┘
                 │
    ┌────────────▼─────────────────┐
    │  3. MODEL PREDICTION         │
    │     Forward pass through LSTM │
    │     Output: [10 × 1]         │
    └────────────┬─────────────────┘
                 │
    ┌────────────▼─────────────────┐
    │  4. INVERSE SCALING          │
    │     Apply target_scaler⁻¹    │
    │     Shape: [0,1] → real values│
    └────────────┬─────────────────┘
                 │
    ┌────────────▼─────────────────┐
    │  5. POST-PROCESSING          │
    │     Clip negative values      │
    │     Final: [queue_t+1...+10] │
    └────────────┬─────────────────┘
                 │
    Output: Next 10 timesteps of predicted queue lengths
    │
    └─► Used by DDQN for state representation


═════════════════════════════════════════════════════════════════════════════════
                      PHASE 7: DDQN AGENT TRAINING
═════════════════════════════════════════════════════════════════════════════════

    Closed-Loop Traffic Signal Control

    ┌─────────────────────────────────────┐
    │  1. STATE REPRESENTATION            │
    │     Current State Features:         │
    │     ├─ Traffic flow                 │
    │     ├─ Queue length                 │
    │     ├─ Signal phase                 │
    │     ├─ Remaining green time         │
    │     ├─ Downstream occupancy         │
    │     └─ ... (9 features total)       │
    │                                      │
    │     Future Predicted Features:      │
    │     ├─ Mean queue length             │
    │     ├─ Max queue length              │
    │     ├─ Mean occupancy                │
    │     └─ ... (aggregated stats)        │
    │                                      │
    │     Total State Dim: ~14-16          │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  2. ACTION SELECTION (ε-greedy)     │
    │     Possible Actions:                │
    │     ├─ Hold current phase            │
    │     ├─ Extend green time (+5 sec)   │
    │     └─ Move to next phase            │
    │     Total: 3 actions                 │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  3. ACTION EXECUTION                 │
    │     Signal Controller updates phase  │
    │     SUMO simulates 60 seconds        │
    │     Collect new observations         │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  4. REWARD COMPUTATION              │
    │     Reward = -(waiting_time)        │
    │              -(queue_length)        │
    │              +(vehicles_passed)     │
    │              -penalty (if invalid)  │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  5. EXPERIENCE REPLAY                │
    │     Store: (s, a, r, s', done)      │
    │     Sample batch of 32 transitions  │
    │     Update Q-Network                 │
    │     Update Target Network (periodic) │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  6. TRAINING LOOP                    │
    │     Episodes: 5 (configurable)       │
    │     Steps/Episode: 30                │
    │     Total Interactions: 150          │
    │     Epsilon decay: Linear            │
    │     ├─ Start: 1.0 (100% random)      │
    │     └─ End: 0.05 (95% greedy)        │
    └─────────────┬───────────────────────┘
                  │
    Output: Trained DDQN agent (ddqn_agent.pth)


═════════════════════════════════════════════════════════════════════════════════
                      PHASE 8: PERFORMANCE EVALUATION
═════════════════════════════════════════════════════════════════════════════════

    Comparison of Control Strategies

    ┌─────────────────────────────────────┐
    │  Baseline: Fixed Signal Timing      │
    │  ├─ Green: 30 sec (fixed)           │
    │  ├─ Red: 30 sec (fixed)             │
    │  └─ No adaptation                   │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Adaptive (LSTM + DDQN)             │
    │  ├─ Uses traffic predictions        │
    │  ├─ Optimizes signal timing         │
    │  └─ Learns from interaction         │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Evaluation Metrics                 │
    │  ├─ Average waiting time            │
    │  ├─ Queue length (peak/avg)         │
    │  ├─ Vehicle throughput              │
    │  ├─ Average speed                   │
    │  ├─ Travel time per vehicle         │
    │  └─ Fuel consumption (optional)     │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │  Results Visualization              │
    │  ├─ Line plots (metrics vs time)    │
    │  ├─ Bar charts (comparison)         │
    │  ├─ Heatmaps (spatial patterns)     │
    │  └─ Summary statistics              │
    └─────────────────────────────────────┘


═════════════════════════════════════════════════════════════════════════════════
                        COMPLETE DATA FLOW SUMMARY
═════════════════════════════════════════════════════════════════════════════════

    SUMO Simulation
            │
            ├─→ [Every 1 second] ─→ TrafficCollector
            │                              │
            │                              ├─→ Extract 13 features
            │                              │
            │       ◄─ Control Signals ◄─ │
            │
            ├─→ [Every 60 seconds] ─→ TrafficPipeline
            │                              │
            │                              ├─→ Aggregate observations
            │                              │
            │                              └─→ CSV Output
            │
            └─→ Dataset (CSV)
                            │
                            ├─→ Clean & Preprocess
                            │
                            ├─→ Generate Sequences
                            │
                            ├─→ Split Train/Test
                            │
                            ├─→ LSTM Training
                            │
                            ├─→ Model Saved (pkl + pth)
                            │
                            ├─→ Real-time Prediction
                            │
                            ├─→ State Vector
                            │
                            ├─→ DDQN Agent
                            │
                            ├─→ Action Selection
                            │
                            └─→ Signal Control
                                    │
                                    └─→ Back to SUMO


═════════════════════════════════════════════════════════════════════════════════
                            MATRIX REPRESENTATION
═════════════════════════════════════════════════════════════════════════════════

    Input Data Shape Transformations:

    1. Raw Collection (per timestep):
       [1 junction × 13 features] = [1 × 13]

    2. Over 30 seconds (lookback window):
       [30 timesteps × 13 features] = [30 × 13]

    3. In batch during training:
       [32 batch × 30 timesteps × 13 features] = [32 × 30 × 13]

    4. After LSTM processing:
       [32 batch × 64 hidden units] = [32 × 64]

    5. After output layer:
       [32 batch × 10 predictions × 1 target] = [32 × 10 × 1]

    6. Final output (single sample):
       [10 future timesteps] = queue_length predictions


═════════════════════════════════════════════════════════════════════════════════
                         OUTPUT FILES GENERATED
═════════════════════════════════════════════════════════════════════════════════

    Directory Structure:

    project_root/
    ├── data/
    │   ├── raw/
    │   │   └─ traffic_observations.csv (raw SUMO data)
    │   └── processed/
    │       └─ location_1_dataset.csv (cleaned & aggregated)
    │
    ├── models/
    │   ├─ lstm_model.pth (trained weights)
    │   ├─ lstm_scalers.pkl (MinMaxScaler objects)
    │   └─ ddqn_agent.pth (trained RL agent)
    │
    ├── outputs/
    │   ├─ training_history.csv (epoch-wise stats)
    │   ├─ metrics.csv (final evaluation metrics)
    │   ├─ predictions.csv (actual vs predicted)
    │   ├─ loss_curve.png (training loss graph)
    │   ├─ prediction_vs_actual.png (validation plot)
    │   ├─ reward_curve.png (DDQN rewards)
    │   └─ performance_comparison.png (baseline vs adaptive)
    │
    └── evaluation/
        └─ final_report.txt (summary results)


═════════════════════════════════════════════════════════════════════════════════
                              KEY INSIGHTS
═════════════════════════════════════════════════════════════════════════════════

    ✓ Data Flow: SUMO → Collector → Pipeline → Dataset → LSTM
    ✓ Prediction: Historical 30 steps → Future 10 steps
    ✓ Integration: LSTM predictions → DDQN state → Signal control
    ✓ Evaluation: Compare fixed vs adaptive signal control
    ✓ Output: Comprehensive metrics and visualization

    Bottlenecks:
    ⚠ Dataset quality (90% queue_length = 0)
    ⚠ Need diverse traffic conditions for better learning
    ⚠ Model checkpoint doesn't include architecture
    ⚠ Scalers must be applied consistently in production

    Future Improvements:
    → Generate heavy traffic scenarios
    → Multi-junction control
    → Real-time model updates
    → Production deployment pipeline

```

---

## 📈 DETAILED MATRIX DIMENSIONS

### Training Pipeline Dimensions

| Stage | Input Shape | Output Shape | Description |
|-------|------------|--------------|-------------|
| Raw Collection | [1] | [13] | Single junction, 13 features |
| Lookback Window | [30, 13] | [30, 13] | 30 timesteps × features |
| Batch | [32, 30, 13] | [32, 30, 13] | Batch of 32 sequences |
| LSTM Layer 1 | [32, 30, 13] | [32, 30, 64] | Process sequence |
| LSTM Layer 2 | [32, 30, 64] | [32, 30, 64] | Stack layers |
| Final Hidden | [32, 30, 64] | [32, 64] | Extract last state |
| Output Layer | [32, 64] | [32, 10] | Generate predictions |
| Reshape | [32, 10] | [32, 10, 1] | Add target dimension |

### Sequence Generation

```
Raw Sequence Index:  0  1  2  3  4  5  6  7  8  9 ... 29 30 31 32 33 34 35 36 37 38 39
                     │──────────── INPUTS (30) ────────────│  │─── TARGETS (10) ───│
                     
                     Sample 1: Input[0:30] → Target[30:40]
                     
                     │──────────── INPUTS (30) ────────────│  │─── TARGETS (10) ───│
                              Sample 2: Input[1:31] → Target[31:41]
                     
                     And so on...
```

---

## 🔄 CONTINUOUS LOOP (Real-Time)

```
While Simulation Running:

    t=0: Collect observations
    t=1: Collect observations
    ...
    t=29: Have 30 observations → Can predict
    
    Prediction 1: t=29 → Predict t=[30:40]
    
    t=30: New observation arrives, oldest drops
    Prediction 2: t=30 → Predict t=[31:41]
    
    t=31: New observation arrives, oldest drops
    Prediction 3: t=31 → Predict t=[32:42]
    
    ... Continue indefinitely in production
```

---

**Generated**: 2026-07-17
**Version**: 1.0
**Status**: Complete Workflow Documentation
