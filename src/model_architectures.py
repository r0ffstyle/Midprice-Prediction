"""
Model architectures for log-return predictions.

The main architectures are:
1. CNN model
2. LSTM model
3. Combined CNN-LSTM model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import logging
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

    
class OptimizedLOBDataset(torch.utils.data.Dataset):
    """
    Optimized PyTorch Dataset for Limit Order Book data.
    """
    
    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 100, # 100 LOB snapshots
        transform: Optional[object] = None,
        is_train: bool = True,
        preload: bool = True
    ):
        """
        Initialize the optimized LOB dataset.
        
        Parameters:
        ----------
        preload : bool, default=True
            Whether to preload all data into memory as tensors
        """
        self.is_train = is_train
        self.seq_length = seq_length
        self.transform = transform
        
        # Convert to numpy arrays if DataFrames
        if hasattr(features, 'values'):
            self.features = features.values.astype(np.float32)
        else:
            self.features = features.astype(np.float32)
            
        if hasattr(targets, 'values'):
            self.targets = targets.values.astype(np.float32)
        else:
            self.targets = targets.astype(np.float32)
        
        # Apply transformation if provided
        if self.transform is not None:
            self.features = self.transform.fit_transform(self.features)
            
        self.feature_dim = self.features.shape[1]
        self.target_dim = self.targets.shape[1]
        
        # Preload data as tensors for faster access
        self.preload = preload
        if preload:
            self.precomputed_sequences = []
            self.precomputed_targets = []
            
            for i in range(len(self)):
                # Extract sequence
                x = self.features[i:i + self.seq_length]
                
                # Get target from last timestamp
                y = self.targets[i + self.seq_length - 1]
                
                # For CNN models, reshape to [channels, seq_length, features]
                x = np.expand_dims(x, axis=0)
                
                # Convert to tensors now to avoid conversion during training
                self.precomputed_sequences.append(torch.tensor(x, dtype=torch.float32))
                self.precomputed_targets.append(torch.tensor(y, dtype=torch.float32))
        
        logger.info(f"Optimized Dataset initialized with {len(self)} sequences")
        
    def __len__(self):
        """Return the number of possible sequences in the dataset."""
        return max(0, len(self.features) - self.seq_length + 1)
    
    def __getitem__(self, idx):
        """Get a sequence of features and its corresponding target."""
        if self.preload:
            # Return precomputed tensors
            return self.precomputed_sequences[idx], self.precomputed_targets[idx]
        else:
            # Extract sequence
            x = self.features[idx:idx + self.seq_length]
            
            # Get target (use the target from the last timestamp in the sequence)
            y = self.targets[idx + self.seq_length - 1]
            
            # For CNN models, reshape to [channels, seq_length, features]
            # This follows the expected input format in Kolm et al.
            x = np.expand_dims(x, axis=0)
            
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class CNN(nn.Module):
    """
    CNN implementation.
    
    The architecture consists of specialized blocks:
    1. Block 1: Fuses price and volume at each level
    2. Block 2: Combines information across bid/ask sides and levels
    3. Block 3: Aggregates information from all levels
    4. Block 4: Inception module with parallel temporal filters
    5. Block 5: Dense layers for prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        num_horizons: int = 10,
        conv_channels: List[int] = [32, 32, 32],
        kernel_sizes: List[int] = [5, 5, 5],  # Not actually used, kept for compatibility
        fc_units: List[int] = [512, 256],
        dropout: float = 0.5
    ):
        super(CNN, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.num_horizons = num_horizons
        
        # Block 1: Combine price and volume at each level
        self.block1 = nn.Sequential(
            # Spatial convolution to fuse price-volume pairs (1×2 kernel)
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels[0],
                kernel_size=(1, 2),
                stride=(1, 2)
            ),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Temporal convolution across the sequence
            nn.Conv2d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[0],
                kernel_size=(4, 1),
                stride=1,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 2: Combine across bid/ask sides at each level
        self.block2 = nn.Sequential(
            # Spatial convolution to combine bid/ask sides (1×2 kernel)
            nn.Conv2d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1],
                kernel_size=(1, 2),
                stride=(1, 2)
            ),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Temporal convolution
            nn.Conv2d(
                in_channels=conv_channels[1],
                out_channels=conv_channels[1],
                kernel_size=(4, 1),
                stride=1,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 3: Aggregate all levels
        # Kernel size depends on how many levels remain after Block 2
        levels = max(1, input_dim // 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels[1],
                out_channels=conv_channels[2],
                kernel_size=(1, levels),  # Combine all remaining levels
                stride=1
            ),
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 4: Inception module with parallel filters
        # Subblock 1: 1×1 conv followed by 3×1 temporal conv
        self.inception_1 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Subblock 2: 1×1 conv followed by 5×1 temporal conv
        self.inception_2 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Subblock 3: Max pooling followed by 1×1 conv
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Calculate the flattened output size after inception module
        with torch.no_grad():
            # Create a dummy input
            x = torch.zeros(1, 1, seq_length, input_dim)
            
            # Pass through blocks 1-3
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            # Process through inception module
            inc1 = self.inception_1(x)
            inc2 = self.inception_2(x)
            inc3 = self.inception_3(x)
            
            # Concatenate and flatten
            x = torch.cat([inc1, inc2, inc3], dim=1)
            self.fc_input_size = x.view(1, -1).size(1)
            
        # Block 5: FC layers
        fc_layers = []
        in_features = self.fc_input_size
        
        for fc_size in fc_units:
            fc_layers.extend([
                nn.Linear(in_features, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = fc_size
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Output layer
        self.output_layer = nn.Linear(in_features, num_horizons)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        logger.info(f"CNN model following Kolm et al. (2023) initialized with {self.count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize weights with specialized initialization for each layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the CNN model following Kolm et al. architecture."""
        batch_size = x.size(0)
        
        # Blocks 1-3: Spatial and temporal convolutions
        x = self.block1(x)  # Fuse price-volume
        x = self.block2(x)  # Combine across bid/ask
        x = self.block3(x)  # Aggregate levels
        
        # Block 4: Inception module
        inc1 = self.inception_1(x)
        inc2 = self.inception_2(x)
        inc3 = self.inception_3(x)
        
        # Concatenate along the channel dimension
        x = torch.cat([inc1, inc2, inc3], dim=1)
        
        # Flatten for FC layers
        x = x.view(batch_size, -1)
        
        # Block 5: FC layers
        x = self.fc_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class LSTM(nn.Module):
    """
    LSTM implementation.
    
    This model processes sequential data through LSTM layers
    without the CNN component from the full architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_horizons: int = 10,
        fc_units: List[int] = [256, 128],
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        self.bidirectional = bidirectional
        
        # LSTM layer with optimized initialization
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output dimension if bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers as a sequential module
        fc_layers = []
        in_features = lstm_output_dim
        
        for fc_size in fc_units:
            fc_layers.extend([
                nn.Linear(in_features, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = fc_size
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Output layer for multi-horizon prediction
        self.output_layer = nn.Linear(in_features, num_horizons)
        
        # Initialize weights for faster convergence
        self._initialize_weights()
        
        logger.info(f"LSTM model initialized with {self.count_parameters()} parameters")

    def _initialize_weights(self):
        """Initialize weights with optimized initialization for faster convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input to hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden to hidden weights (recurrent)
                nn.init.orthogonal_(param)  # Orthogonal init for recurrent connections
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif isinstance(param, nn.Linear):
                nn.init.kaiming_normal_(param.weight, mode='fan_out', nonlinearity='relu')
                if param.bias is not None:
                    nn.init.constant_(param.bias, 0)

    def forward(self, x):
        """Forward pass through the LSTM model."""
        # Input shape: [batch_size, 1, seq_length, input_dim]
        # Reshape for LSTM: [batch_size, seq_length, input_dim]
        x = x.squeeze(1).contiguous()
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output from the sequence
        x = lstm_out[:, -1]
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class CNN_LSTM(nn.Module):
    """
    CNN-LSTM implementation.
    
    The architecture consists of specialized blocks:
    1. Block 1: Fuses price and volume at each level
    2. Block 2: Combines information across bid/ask sides and levels
    3. Block 3: Aggregates information from all levels
    4. Block 4: Inception module with parallel temporal filters
    5. Block 5: LSTM and dense layers for prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        num_horizons: int = 10,
        conv_channels: List[int] = [32, 32, 32],
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        fc_units: List[int] = [256, 128],
        dropout: float = 0.5
    ):
        super(CNN_LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.num_horizons = num_horizons
        
        # Block 1: Combine price and volume at each level
        self.block1 = nn.Sequential(
            # Spatial convolution to fuse price-volume pairs (1×2 kernel)
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels[0],
                kernel_size=(1, 2),
                stride=(1, 2)
            ),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Temporal convolution across the sequence
            nn.Conv2d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[0],
                kernel_size=(4, 1),
                stride=1,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 2: Combine across bid/ask sides at each level
        self.block2 = nn.Sequential(
            # Spatial convolution to combine bid/ask sides (1×2 kernel)
            nn.Conv2d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1],
                kernel_size=(1, 2),
                stride=(1, 2)
            ),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Temporal convolution
            nn.Conv2d(
                in_channels=conv_channels[1],
                out_channels=conv_channels[1],
                kernel_size=(4, 1),
                stride=1,
                padding=(2, 0)
            ),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 3: Aggregate all levels
        # Kernel size depends on how many levels remain after Block 2
        # For 10 levels reduced by 2× in Block 1 and 2× in Block 2, we have 10/4 = 2.5 -> use kernel=(1, max(1, input_dim // 4))
        levels = max(1, input_dim // 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels[1],
                out_channels=conv_channels[2],
                kernel_size=(1, levels),  # Combine all remaining levels
                stride=1
            ),
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Block 4: Inception module with parallel filters
        # Subblock 1: 1×1 conv followed by 3×1 temporal conv
        self.inception_1 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Subblock 2: 1×1 conv followed by 5×1 temporal conv
        self.inception_2 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Subblock 3: Max pooling followed by 1×1 conv
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Calculate the output size after inception module
        with torch.no_grad():
            # Create a dummy input
            x = torch.zeros(1, 1, seq_length, input_dim)
            
            # Pass through blocks 1-3
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            # Process through inception module
            inc1 = self.inception_1(x)
            inc2 = self.inception_2(x)
            inc3 = self.inception_3(x)
            
            # Concatenate and reshape for LSTM
            x = torch.cat([inc1, inc2, inc3], dim=1)
            x = x.permute(0, 2, 1, 3)  # [batch, seq, channels, features]
            self.lstm_input_size = x.size(2) * x.size(3)  # channels * features
            self.lstm_seq_len = x.size(1)  # sequence length
            
        # Block 5: LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # FC layers
        fc_layers = []
        in_features = lstm_hidden_dim
        
        for fc_size in fc_units:
            fc_layers.extend([
                nn.Linear(in_features, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = fc_size
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Output layer
        self.output_layer = nn.Linear(in_features, num_horizons)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        logger.info(f"CNN-LSTM model following Kolm et al. (2023) initialized with {self.count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize weights with specialized initialization for each layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # LSTM-specific initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)  # Orthogonal init for recurrent connections
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """Forward pass through the CNN-LSTM model following Kolm et al. architecture."""
        batch_size = x.size(0)
        
        # Blocks 1-3: Spatial and temporal convolutions
        x = self.block1(x)  # Fuse price-volume
        x = self.block2(x)  # Combine across bid/ask
        x = self.block3(x)  # Aggregate levels
        
        # Block 4: Inception module
        inc1 = self.inception_1(x)
        inc2 = self.inception_2(x)
        inc3 = self.inception_3(x)
        
        # Concatenate along the channel dimension
        x = torch.cat([inc1, inc2, inc3], dim=1)
        
        # Reshape for LSTM: [batch, seq, channels, features] -> [batch, seq, channels*features]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.lstm_seq_len, self.lstm_input_size)
        
        # Block 5: LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output from the sequence
        x = lstm_out[:, -1]
        
        # FC layers
        x = self.fc_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class CNN_Transformer(nn.Module):
    """CNN + Transformer Encoder for high-frequency LOB forecasting."""
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        num_horizons: int = 10,
        conv_channels: List[int] = [32, 32, 32],
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super(CNN_Transformer, self).__init__()
        # CNN Blocks (same as CNN_LSTM)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_channels[0], conv_channels[0], kernel_size=(4, 1), padding=(2, 0)),
            nn.BatchNorm2d(conv_channels[0]),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_channels[1], conv_channels[1], kernel_size=(4, 1), padding=(2, 0)),
            nn.BatchNorm2d(conv_channels[1]),
            nn.LeakyReLU(0.01, inplace=True)
        )
        levels = max(1, input_dim // 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=(1, levels)),
            nn.BatchNorm2d(conv_channels[2]),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Inception Module
        self.inception_1 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.inception_2 = nn.Sequential(
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(conv_channels[2], 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Determine sequence and feature dims after CNN + Inception
        with torch.no_grad():
            x = torch.zeros(1, 1, seq_length, input_dim)
            x = self.block1(x); x = self.block2(x); x = self.block3(x)
            inc1 = self.inception_1(x); inc2 = self.inception_2(x); inc3 = self.inception_3(x)
            x = torch.cat([inc1, inc2, inc3], dim=1)
            x = x.permute(0, 2, 1, 3)
            batch, seq, ch, feat = x.size()
            self.seq_len = seq
            self.feat_dim = ch * feat
        # Transformer-specific layers
        self.input_proj = nn.Linear(self.feat_dim, d_model)
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(self.seq_len, d_model), requires_grad=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_horizons)
        )
        self._initialize_weights()
        logger.info(f"CNN-Transformer model initialized with {self.count_parameters()} parameters")

    def _generate_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (seq_len, 1, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        # CNN + Inception
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        inc1 = self.inception_1(x); inc2 = self.inception_2(x); inc3 = self.inception_3(x)
        x = torch.cat([inc1, inc2, inc3], dim=1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.seq_len, -1)
        # Project and add positional
        x = self.input_proj(x) + self.positional_encoding[:self.seq_len].transpose(0, 1)
        # Transformer expects (seq, batch, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # Take last time step
        x = x[-1]
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create an optimized model instance.
    
    Parameters:
    ----------
    model_type : str
        Type of model to create ('cnn', 'lstm', or 'cnn_lstm')
    **kwargs : dict
        Additional arguments for the model constructor
        
    Returns:
    -------
    nn.Module
        Instantiated optimized model
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return CNN(**kwargs)
    elif model_type == 'lstm':
        return LSTM(**kwargs)
    elif model_type == 'cnn_lstm':
        return CNN_LSTM(**kwargs)
    elif model_type == 'cnn_transformer':
        return CNN_Transformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")