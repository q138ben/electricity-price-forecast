import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class ElectricityDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_col: str, sequence_length: int = 24):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_col : str
            Name of the target column
        sequence_length : int
            Number of time steps to use for prediction
        """
        self.data = data
        self.target_col = target_col
        self.sequence_length = sequence_length
        
        # Separate features and target
        self.features = data.drop(columns=[target_col])
        self.target = data[target_col].values  # Convert to numpy array
        
        # Scale features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - 24  # Ensure we have enough data for target
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence of features
        X = torch.FloatTensor(self.features_scaled[idx:idx + self.sequence_length])
        
        # Get target (next 24 hours)
        y = torch.FloatTensor(self.target[idx + self.sequence_length:idx + self.sequence_length + 24])
        
        return X, y

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the LSTM model
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 24)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Forward propagate through fully connected layers
        out = self.fc(out)  # Shape: (batch_size, 24)
        
        return out

def train_lstm(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               num_epochs: int,
               device: torch.device,
               save_dir: str) -> Dict[str, List[float]]:
    """
    Train the LSTM model
    
    Parameters:
    -----------
    model : nn.Module
        LSTM model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    num_epochs : int
        Number of epochs to train
    device : torch.device
        Device to train on
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    Dict[str, List[float]]
        Training history
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
    
    return history

def predict_lstm(model: nn.Module,
                test_loader: DataLoader,
                device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained LSTM model
    
    Parameters:
    -----------
    model : nn.Module
        Trained LSTM model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to make predictions on
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Predictions and actual values
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.numpy())
    
    return np.array(predictions), np.array(actuals)

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Plot training history
    
    Parameters:
    -----------
    history : Dict[str, List[float]]
        Training history
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(data: pd.DataFrame,
                      target_col: str,
                      sequence_length: int = 24,
                      batch_size: int = 32,
                      num_epochs: int = 100,
                      save_dir: str = 'models/lstm') -> Dict[str, float]:
    """
    Train and evaluate the LSTM model
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
    sequence_length : int
        Number of time steps to use for prediction
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of epochs to train
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset
    dataset = ElectricityDataset(data, target_col, sequence_length)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, len(dataset) - train_size - val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    input_size = len(data.columns) - 1  # Exclude target column
    model = LSTMModel(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    history = train_lstm(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, save_dir
    )
    
    # Plot training history
    plot_training_history(history, f'{save_dir}/training_history.png')
    
    # Load best model and make predictions
    model.load_state_dict(torch.load(f'{save_dir}/best_model.pth'))
    predictions, actuals = predict_lstm(model, test_loader, device)
    
    # Calculate metrics
    metrics = {
        'mse': np.mean((predictions - actuals) ** 2),
        'mae': np.mean(np.abs(predictions - actuals)),
        'rmse': np.sqrt(np.mean((predictions - actuals) ** 2))
    }
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(f'{save_dir}/metrics.csv', index=False)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    from data_preparation import load_hourly_data_split
    
    # Load data
    hourly_data = load_hourly_data_split('data')
    
    # Train and evaluate for each hour
    for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
        print(f"\nTraining LSTM model for hour {hour}")
        
        # Combine features and target
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Train and evaluate
        metrics = train_and_evaluate(
            train_data,
            target_col='target_price',
            save_dir=f'models/lstm/hour_{hour}'
        )
        
        print(f"Metrics for hour {hour}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}") 