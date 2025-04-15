"""
Model Module

This module defines the LSTM-based models for stock price prediction,
including attention mechanisms and multi-step forecasting capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    """
    Attention mechanism layer for LSTM.
    """
    def __init__(self, hidden_dim):
        """
        Initialize the attention layer.

        Parameters:
        -----------
        hidden_dim : int
            Dimension of the hidden state
        """
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Forward pass of the attention layer.

        Parameters:
        -----------
        lstm_output : torch.Tensor
            Output from LSTM layer with shape (batch_size, seq_length, hidden_dim)

        Returns:
        --------
        tuple
            (context_vector, attention_weights)
            context_vector has shape (batch_size, hidden_dim)
            attention_weights has shape (batch_size, seq_length, 1)
        """
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)

        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector, attention_weights

class LSTMModel(nn.Module):
    """
    Basic LSTM model for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        """
        Initialize the LSTM model.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        dropout_prob : float, optional
            Dropout probability
        """
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, output_dim)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(lstm_out)

        # Apply output layer
        out = self.fc(out)

        return out

class LSTMWithAttention(nn.Module):
    """
    LSTM model with attention mechanism for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        """
        Initialize the LSTM model with attention.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        dropout_prob : float, optional
            Dropout probability
        """
        super(LSTMWithAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM model with attention.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        tuple
            (output, attention_weights)
            output has shape (batch_size, output_dim)
            attention_weights has shape (batch_size, seq_length, 1)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)

        # Apply dropout
        out = self.dropout(context_vector)

        # Apply output layer
        out = self.fc(out)

        return out, attention_weights

class StackedLSTMWithAttention(nn.Module):
    """
    Stacked LSTM model with attention and residual connections for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        """
        Initialize the stacked LSTM model with attention and residual connections.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        dropout_prob : float, optional
            Dropout probability
        """
        super(StackedLSTMWithAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Additional LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        for i in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            ))

        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the stacked LSTM model with attention and residual connections.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        tuple
            (output, attention_weights)
            output has shape (batch_size, output_dim)
            attention_weights has shape (batch_size, seq_length, 1)
        """
        # First LSTM layer
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropouts[0](out)

        # Additional LSTM layers with residual connections
        for i in range(1, self.num_layers):
            residual = out
            out, _ = self.lstm_layers[i-1](out)
            out = self.dropouts[i](out)
            out = out + residual  # Residual connection

        # Apply attention
        context_vector, attention_weights = self.attention(out)

        # Apply output layer
        out = self.fc(context_vector)

        return out, attention_weights

class MultiStepLSTM(nn.Module):
    """
    LSTM model for multi-step forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, forecast_horizon, dropout_prob=0.2):
        """
        Initialize the multi-step LSTM model.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        forecast_horizon : int
            Number of steps ahead to forecast
        dropout_prob : float, optional
            Dropout probability
        """
        super(MultiStepLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer for multi-step forecasting
        self.fc = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x):
        """
        Forward pass of the multi-step LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, forecast_horizon)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(lstm_out)

        # Apply output layer
        out = self.fc(out)

        return out

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        """
        Initialize the Bidirectional LSTM model.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        dropout_prob : float, optional
            Dropout probability
        """
        super(BidirectionalLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer (note: hidden_dim * 2 because of bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass of the Bidirectional LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, output_dim)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(lstm_out)

        # Apply output layer
        out = self.fc(out)

        return out

class ConvLSTM(nn.Module):
    """
    Convolutional LSTM model for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, kernel_size=3, dropout_prob=0.2):
        """
        Initialize the Convolutional LSTM model.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        kernel_size : int, optional
            Size of the convolutional kernel
        dropout_prob : float, optional
            Dropout probability
        """
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # 1D Convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2  # Same padding
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Convolutional LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, output_dim)
        """
        # Reshape for 1D convolution [batch, channels, length]
        x_conv = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_length]

        # Apply 1D convolution
        x_conv = self.conv1d(x_conv)  # [batch_size, hidden_dim, seq_length]

        # Reshape back for LSTM [batch, seq_length, features]
        x_conv = x_conv.permute(0, 2, 1)  # [batch_size, seq_length, hidden_dim]

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x_conv, (h0, c0))

        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(lstm_out)

        # Apply output layer
        out = self.fc(out)

        return out

class TransformerModel(nn.Module):
    """
    Transformer model for stock price prediction.
    """
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, output_dim, dropout_prob=0.2):
        """
        Initialize the Transformer model.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Dimension of the hidden state
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        output_dim : int
            Number of output dimensions (1 for single-step, >1 for multi-step)
        dropout_prob : float, optional
            Dropout probability
        """
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout_prob)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim*4, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Transformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, output_dim)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_length, hidden_dim]

        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_length, hidden_dim]

        # Reshape for transformer [seq_length, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [seq_length, batch_size, hidden_dim]

        # Get the output from the last time step
        x = x[-1, :, :]  # [batch_size, hidden_dim]

        # Apply dropout
        x = self.dropout(x)

        # Apply output layer
        x = self.fc(x)  # [batch_size, output_dim]

        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, d_model)

        Returns:
        --------
        torch.Tensor
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

def create_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2, **kwargs):
    """
    Create a model based on the specified type.

    Parameters:
    -----------
    model_type : str
        Type of model to create. Options: 'lstm', 'lstm_attention', 'stacked_lstm_attention', 'multi_step_lstm',
        'bidirectional_lstm', 'conv_lstm', 'transformer'
    input_dim : int
        Number of input features
    hidden_dim : int
        Dimension of the hidden state
    num_layers : int
        Number of LSTM layers
    output_dim : int
        Number of output dimensions (1 for single-step, >1 for multi-step)
    dropout_prob : float, optional
        Dropout probability
    **kwargs : dict
        Additional keyword arguments for specific model types:
        - kernel_size : int (for 'conv_lstm')
        - nhead : int (for 'transformer')

    Returns:
    --------
    torch.nn.Module
        The created model
    """
    try:
        logger.info(f"Creating {model_type} model")

        if model_type == 'lstm':
            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        elif model_type == 'lstm_attention':
            model = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        elif model_type == 'stacked_lstm_attention':
            model = StackedLSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        elif model_type == 'multi_step_lstm':
            model = MultiStepLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        elif model_type == 'bidirectional_lstm':
            model = BidirectionalLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        elif model_type == 'conv_lstm':
            kernel_size = kwargs.get('kernel_size', 3)
            model = ConvLSTM(input_dim, hidden_dim, num_layers, output_dim, kernel_size, dropout_prob)
        elif model_type == 'transformer':
            nhead = kwargs.get('nhead', 8)  # Default to 8 attention heads
            model = TransformerModel(input_dim, hidden_dim, nhead, num_layers, output_dim, dropout_prob)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_dim = 10  # Number of features
    hidden_dim = 64  # Hidden dimension
    num_layers = 2  # Number of LSTM layers
    output_dim = 1  # Single-step forecasting

    # Create a batch of data
    batch_size = 32
    seq_length = 20
    x = torch.randn(batch_size, seq_length, input_dim)

    # Create models
    lstm_model = create_model('lstm', input_dim, hidden_dim, num_layers, output_dim)
    lstm_attention_model = create_model('lstm_attention', input_dim, hidden_dim, num_layers, output_dim)
    stacked_lstm_attention_model = create_model('stacked_lstm_attention', input_dim, hidden_dim, num_layers, output_dim)
    multi_step_lstm_model = create_model('multi_step_lstm', input_dim, hidden_dim, num_layers, 5)  # 5-step forecast
    bidirectional_lstm_model = create_model('bidirectional_lstm', input_dim, hidden_dim, num_layers, output_dim)
    conv_lstm_model = create_model('conv_lstm', input_dim, hidden_dim, num_layers, output_dim, kernel_size=3)
    transformer_model = create_model('transformer', input_dim, hidden_dim, num_layers, output_dim, nhead=8)

    # Test models
    lstm_output = lstm_model(x)
    lstm_attention_output, attention_weights = lstm_attention_model(x)
    stacked_lstm_attention_output, stacked_attention_weights = stacked_lstm_attention_model(x)
    multi_step_output = multi_step_lstm_model(x)
    bidirectional_output = bidirectional_lstm_model(x)
    conv_lstm_output = conv_lstm_model(x)
    transformer_output = transformer_model(x)

    print(f"LSTM output shape: {lstm_output.shape}")
    print(f"LSTM with attention output shape: {lstm_attention_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Stacked LSTM with attention output shape: {stacked_lstm_attention_output.shape}")
    print(f"Multi-step LSTM output shape: {multi_step_output.shape}")
    print(f"Bidirectional LSTM output shape: {bidirectional_output.shape}")
    print(f"Conv LSTM output shape: {conv_lstm_output.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")

    # Print model parameters
    print("\nModel parameters:")
    print(f"LSTM: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"LSTM with attention: {sum(p.numel() for p in lstm_attention_model.parameters()):,}")
    print(f"Stacked LSTM with attention: {sum(p.numel() for p in stacked_lstm_attention_model.parameters()):,}")
    print(f"Multi-step LSTM: {sum(p.numel() for p in multi_step_lstm_model.parameters()):,}")
    print(f"Bidirectional LSTM: {sum(p.numel() for p in bidirectional_lstm_model.parameters()):,}")
    print(f"Conv LSTM: {sum(p.numel() for p in conv_lstm_model.parameters()):,}")
    print(f"Transformer: {sum(p.numel() for p in transformer_model.parameters()):,}")
