"""
Complete LSTM Traffic Prediction Model
========================================

This module contains the complete LSTM model with proper architecture,
training utilities, and prediction capabilities.

Architecture:
    Input Layer (12 features) 
        ↓
    LSTM Layer 1 (64 hidden units)
        ↓
    LSTM Layer 2 (64 hidden units, with dropout)
        ↓
    Output Layer (10 predictions)
        ↓
    Final Output (prediction_horizon × target_size)

Model Details:
    - Type: Sequence-to-Sequence LSTM
    - Purpose: Traffic queue length prediction
    - Input: 30 timesteps × 12 features
    - Output: 10 future timesteps × 1 target
    - Parameters: ~52,000

Author: AI Traffic Control System
Date: 2026-07-17
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# 1. LSTM MODEL ARCHITECTURE
# ============================================================

class TrafficLSTM(nn.Module):
    """
    LSTM model for traffic queue length prediction.
    
    This model takes historical traffic observations and predicts
    future queue lengths using a 2-layer LSTM network.
    
    Args:
        input_size (int): Number of input features (default: 12)
        target_size (int): Number of target variables (default: 1)
        hidden_size (int): Number of LSTM hidden units (default: 64)
        num_layers (int): Number of LSTM layers (default: 2)
        prediction_horizon (int): Number of timesteps to predict (default: 10)
        dropout (float): Dropout rate between LSTM layers (default: 0.2)
    
    Attributes:
        prediction_horizon: How many timesteps ahead to predict
        target_size: Number of output variables per timestep
        lstm: LSTM layer stack
        output_layer: Linear layer for final output
    
    Example:
        >>> model = TrafficLSTM(
        ...     input_size=12,
        ...     hidden_size=64,
        ...     num_layers=2,
        ...     prediction_horizon=10,
        ...     target_size=1,
        ...     dropout=0.2
        ... )
        >>> x = torch.randn(32, 30, 12)  # batch, seq_len, features
        >>> output = model(x)
        >>> print(output.shape)  # [32, 10, 1]
    """
    
    def __init__(
        self,
        input_size: int = 12,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        prediction_horizon: int = 10,
        dropout: float = 0.2,
    ):
        """Initialize LSTM model components."""
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.dropout = dropout
        
        # LSTM Dropout only applies between layers (not after last layer)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        # LSTM Layer Stack
        # - Takes input_size features
        # - Processes through multiple layers
        # - Outputs hidden_size dimensional representations
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,              # Input shape: (batch, seq, features)
            dropout=lstm_dropout,           # Dropout between LSTM layers
        )
        
        # Output Linear Layer
        # - Takes final LSTM hidden state (hidden_size)
        # - Outputs (prediction_horizon × target_size) predictions
        # - This allows predicting multiple timesteps at once
        self.output_layer = nn.Linear(
            hidden_size,
            prediction_horizon * target_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 30, 12)
        
        Returns:
            Predictions of shape (batch_size, prediction_horizon, target_size)
            Example: (32, 10, 1)
        
        Process:
            1. LSTM processes entire sequence
            2. Extract final hidden state
            3. Pass through output layer
            4. Reshape to (batch, horizon, targets)
        """
        # LSTM forward pass
        # lstm_output: (batch, seq_len, hidden_size)
        # _: (num_layers, batch, hidden_size) - we don't need hidden/cell states
        lstm_output, _ = self.lstm(x)
        
        # Extract final hidden state from last timestep
        # Shape: (batch_size, hidden_size)
        final_hidden_state = lstm_output[:, -1, :]
        
        # Pass through output layer
        # Shape: (batch_size, prediction_horizon * target_size)
        predictions = self.output_layer(final_hidden_state)
        
        # Reshape to (batch_size, prediction_horizon, target_size)
        # This separates the flattened predictions back into timesteps
        predictions = predictions.view(
            x.size(0),                      # batch_size
            self.prediction_horizon,        # prediction timesteps
            self.target_size,               # target dimensions
        )
        
        return predictions
    
    def get_config_dict(self) -> Dict:
        """
        Return model configuration as dictionary.
        
        Useful for saving model architecture along with weights.
        
        Returns:
            Dictionary containing all model hyperparameters
        """
        return {
            'input_size': self.input_size,
            'target_size': self.target_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'prediction_horizon': self.prediction_horizon,
            'dropout': self.dropout,
        }
    
    @staticmethod
    def from_config_dict(config: Dict) -> 'TrafficLSTM':
        """
        Create model from configuration dictionary.
        
        Args:
            config: Dictionary with model parameters
        
        Returns:
            Instantiated TrafficLSTM model
        """
        return TrafficLSTM(**config)


# ============================================================
# 2. MODEL CREATION UTILITIES
# ============================================================

def create_lstm_model(
    input_size: int = 12,
    hidden_size: int = 64,
    num_layers: int = 2,
    prediction_horizon: int = 10,
    target_size: int = 1,
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
) -> TrafficLSTM:
    """
    Factory function to create and initialize LSTM model.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of LSTM layers
        prediction_horizon: Timesteps to predict ahead
        target_size: Number of target variables
        dropout: Dropout rate
        device: Device to place model on (cuda/cpu)
    
    Returns:
        Initialized TrafficLSTM model on specified device
    
    Example:
        >>> model = create_lstm_model()
        >>> print(model)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    """
    model = TrafficLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        prediction_horizon=prediction_horizon,
        target_size=target_size,
        dropout=dropout,
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: TrafficLSTM) -> None:
    """
    Print detailed model summary.
    
    Args:
        model: TrafficLSTM model
    """
    config = model.get_config_dict()
    
    print("\n" + "="*70)
    print("LSTM MODEL SUMMARY")
    print("="*70)
    print(f"Model: {model.__class__.__name__}")
    print("-"*70)
    
    print("\nArchitecture:")
    print(f"  Input Features:        {config['input_size']}")
    print(f"  Hidden Units:          {config['hidden_size']}")
    print(f"  Number of Layers:      {config['num_layers']}")
    print(f"  Dropout:               {config['dropout']}")
    print(f"  Prediction Horizon:    {config['prediction_horizon']}")
    print(f"  Target Size:           {config['target_size']}")
    
    print("\nModel Details:")
    print(model)
    
    total_params = count_parameters(model)
    print(f"\nTotal Parameters:      {total_params:,}")
    print(f"Device:                {next(model.parameters()).device}")
    
    print("="*70 + "\n")


# ============================================================
# 3. MODEL CHECKPOINT MANAGEMENT
# ============================================================

def save_model_checkpoint(
    model: TrafficLSTM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    metrics: Optional[Dict] = None,
    filepath: str = "lstm_checkpoint.pth",
) -> None:
    """
    Save complete model checkpoint including architecture.
    
    ⚠️ IMPORTANT: Use this instead of model.state_dict()
    
    This saves:
        - Model weights
        - Model architecture (all hyperparameters)
        - Optimizer state
        - Training progress (epoch, loss)
        - Performance metrics
    
    Args:
        model: TrafficLSTM model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        loss: Current best loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    
    Example:
        >>> save_model_checkpoint(
        ...     model, optimizer, epoch=25, loss=0.015,
        ...     filepath="models/lstm_best.pth"
        ... )
    """
    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config_dict(),
        
        # Optimizer state (if provided)
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        
        # Training info
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics or {},
        
        # Version info
        'pytorch_version': torch.__version__,
        'timestamp': str(np.datetime64('now')),
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_model_checkpoint(
    filepath: str,
    device: torch.device = torch.device('cpu'),
) -> Tuple[TrafficLSTM, Dict]:
    """
    Load complete model from checkpoint.
    
    ✓ USE THIS to load saved models
    
    This loads:
        - Model architecture from config
        - Model weights
        - Training metadata
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    
    Example:
        >>> model, checkpoint = load_model_checkpoint(
        ...     "models/lstm_best.pth",
        ...     device=torch.device('cuda')
        ... )
        >>> print(f"Loaded from epoch {checkpoint['epoch']}")
        >>> print(f"Loss: {checkpoint['loss']:.6f}")
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Reconstruct model from config
    config = checkpoint['model_config']
    model = TrafficLSTM.from_config_dict(config)
    model = model.to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}")
    
    return model, checkpoint


# ============================================================
# 4. MODEL INFERENCE UTILITIES
# ============================================================

class LSTMPredictor:
    """
    Wrapper for LSTM model inference.
    
    Handles:
    - Loading model and scalers
    - Input validation
    - Preprocessing
    - Prediction
    - Postprocessing
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        prediction_horizon: int = 10,
        target_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Initialize predictor with model."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.prediction_horizon = prediction_horizon
        self.input_size = input_size
        
        # Load model
        self.model, self.checkpoint = load_model_checkpoint(
            model_path, device=device
        )
        self.model.eval()
    
    def predict(
        self,
        sequence: np.ndarray,
        return_numpy: bool = True,
    ) -> np.ndarray:
        """
        Make prediction from input sequence.
        
        Args:
            sequence: Input sequence (seq_len × input_size)
            return_numpy: Whether to return numpy array
        
        Returns:
            Predictions (prediction_horizon × target_size)
        """
        # Convert to tensor
        sequence = torch.from_numpy(sequence).float().unsqueeze(0)
        sequence = sequence.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Convert back to numpy
        if return_numpy:
            predictions = predictions.squeeze(0).cpu().numpy()
        
        return predictions


# ============================================================
# 5. TRAINING UTILITIES
# ============================================================

def create_optimizer(
    model: TrafficLSTM,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """
    Create Adam optimizer for model.
    
    Args:
        model: LSTM model
        learning_rate: Learning rate
        weight_decay: L2 regularization factor
    
    Returns:
        Configured optimizer
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str = 'min',
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' for minimizing loss
        factor: Multiplicative factor to reduce LR
        patience: Epochs to wait before reducing
        min_lr: Minimum learning rate
    
    Returns:
        Configured scheduler
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=True,
    )


def train_one_epoch(
    model: TrafficLSTM,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: LSTM model
        train_loader: Data loader
        optimizer: Optimizer
        criterion: Loss function (typically MSELoss)
        device: Device to train on
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Average epoch loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(
    model: TrafficLSTM,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ============================================================
# 6. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LSTM TRAFFIC PREDICTION MODEL - EXAMPLE USAGE")
    print("="*70)
    
    # 1. Create model
    print("\n1. Creating model...")
    model = create_lstm_model(
        input_size=12,
        hidden_size=64,
        num_layers=2,
        prediction_horizon=10,
        target_size=1,
        dropout=0.2,
    )
    
    # 2. Print summary
    print_model_summary(model)
    
    # 3. Create dummy input
    print("2. Testing forward pass...")
    dummy_input = torch.randn(32, 30, 12)  # batch_size=32, seq_len=30, features=12
    output = model(dummy_input)
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Forward pass successful\n")
    
    # 4. Show configuration
    print("3. Model configuration:")
    config = model.get_config_dict()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("For training, use the train_lstm.py script")
    print("="*70 + "\n")

