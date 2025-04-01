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
        self.data = data.copy()
        self.target_col = target_col
        self.sequence_length = sequence_length
        
        # Print initial data info
        print("\nInitial data info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")
        print(f"Null values:\n{self.data.isnull().sum()}")
        
        # Convert all column names to strings
        self.data.columns = self.data.columns.astype(str)
        
        # Handle missing values using newer pandas syntax
        self.data = self.data.ffill().bfill()
        
        # Check for remaining null values
        null_cols = self.data.columns[self.data.isnull().any()].tolist()
        if null_cols:
            print("\nWarning: Null values found in columns:", null_cols)
            print("Attempting to handle remaining null values...")
            
            # Try to fill remaining nulls with column means
            for col in null_cols:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
                else:
                    # For non-numeric columns, fill with most frequent value
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        # Final null check
        if self.data.isnull().any().any():
            null_cols = self.data.columns[self.data.isnull().any()].tolist()
            raise ValueError(f"Unable to handle null values in columns: {null_cols}")
        
        # Check for infinite values
        inf_cols = self.data.columns[np.isinf(self.data).any()].tolist()
        if inf_cols:
            print("\nWarning: Infinite values found in columns:", inf_cols)
            print("Replacing infinite values with NaN...")
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.ffill().bfill()
        
        # Separate features and target
        self.features = self.data.drop(columns=[target_col])
        self.target = self.data[target_col].values
        
        # Validate target values
        if np.isnan(self.target).any() or np.isinf(self.target).any():
            raise ValueError("Target values contain invalid values")
        
        # Scale features
        self.feature_scaler = StandardScaler()
        self.features_scaled = self.feature_scaler.fit_transform(self.features)
        
        # Validate scaled features
        if np.isnan(self.features_scaled).any() or np.isinf(self.features_scaled).any():
            raise ValueError("Feature scaling produced invalid values")
        
        # Scale target
        self.target_scaler = StandardScaler()
        self.target_scaled = self.target_scaler.fit_transform(self.target.reshape(-1, 1)).flatten()
        
        # Validate scaled target
        if np.isnan(self.target_scaled).any() or np.isinf(self.target_scaled).any():
            raise ValueError("Target scaling produced invalid values")
        
        print(f"\nDataset created successfully:")
        print(f"Number of samples: {len(self)}")
        print(f"Feature shape: {self.features_scaled.shape}")
        print(f"Target shape: {self.target_scaled.shape}")
        print(f"Feature columns: {self.features.columns.tolist()}")
        print(f"Target column: {self.target_col}")
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - 24
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence of features
        X = torch.FloatTensor(self.features_scaled[idx:idx + self.sequence_length])
        
        # Get target (next 24 hours)
        y = torch.FloatTensor(self.target_scaled[idx + self.sequence_length:idx + self.sequence_length + 24])
        
        # Validate tensors
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError(f"Invalid feature values at index {idx}")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError(f"Invalid target values at index {idx}")
        
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
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 24)
        )
        
        # Initialize FC weights
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output
        out = out[:, -1, :]
        
        # Forward propagate through fully connected layers
        out = self.fc(out)
        
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
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}")
                continue
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
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
        if val_loss < best_val_loss and not np.isnan(val_loss):
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
    
    # Inverse transform predictions and actuals
    predictions = dataset.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = dataset.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
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
    
    # Combine all data
    all_X_train = pd.DataFrame()
    all_X_test = pd.DataFrame()
    all_y_train = pd.Series()
    all_y_test = pd.Series()
    
    for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
        # Ensure column names are strings and unique
        X_train.columns = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_test.columns = [f'feature_{i}' for i in range(X_test.shape[1])]
        
        # Add hour information
        X_train['hour'] = hour
        X_test['hour'] = hour
        
        # Ensure y_train and y_test are Series with proper index
        y_train = pd.Series(y_train.values.flatten(), index=X_train.index)
        y_test = pd.Series(y_test.values.flatten(), index=X_test.index)
        
        all_X_train = pd.concat([all_X_train, X_train])
        all_X_test = pd.concat([all_X_test, X_test])
        all_y_train = pd.concat([all_y_train, y_train])
        all_y_test = pd.concat([all_y_test, y_test])
    
    # Reset indices
    all_X_train = all_X_train.reset_index(drop=True)
    all_X_test = all_X_test.reset_index(drop=True)
    all_y_train = all_y_train.reset_index(drop=True)
    all_y_test = all_y_test.reset_index(drop=True)
    
    # Verify shapes match
    print("\nData shapes before combination:")
    print(f"X_train shape: {all_X_train.shape}")
    print(f"y_train shape: {all_y_train.shape}")
    print(f"X_test shape: {all_X_test.shape}")
    print(f"y_test shape: {all_y_test.shape}")
    
    # Combine features and target
    train_data = pd.concat([all_X_train, pd.DataFrame({'target_price': all_y_train})], axis=1)
    test_data = pd.concat([all_X_test, pd.DataFrame({'target_price': all_y_test})], axis=1)
    
    print("\nTraining single LSTM model for all hours")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Columns: {train_data.columns.tolist()}")
    print("\nSample of training data:")
    print(train_data.head())
    print("\nData types:")
    print(train_data.dtypes)
    
    # Train and evaluate
    metrics = train_and_evaluate(
        train_data,
        target_col='target_price',
        sequence_length=24,  # Use 24 hours of data to predict next 24 hours
        batch_size=64,  # Increased batch size since we have more data
        num_epochs=100,
        save_dir='models/lstm'
    )
    
    print("\nOverall Model Performance:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}") 