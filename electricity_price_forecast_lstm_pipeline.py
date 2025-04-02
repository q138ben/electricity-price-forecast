import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers.legacy import Adam

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the electricity price data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Rename columns if they have spaces or special characters
    df.columns = [col.strip() for col in df.columns]
    
    # Convert HourUTC to datetime and set as index
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])
    df = df.set_index('HourUTC')
    
    # Sort by datetime
    df = df.sort_index()
    
    # Check for and handle any missing values
    if df.isnull().sum().sum() > 0:
        df = df.interpolate(method='time')
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df.index.dayofweek >= 5
    
    return df

# 2. Feature Engineering and Sequence Creation
def prepare_lstm_data(df, price_col='SpotPriceDKK', lookback=168, forecast_horizon=24):
    """
    Prepare data for LSTM model with a specified lookback period
    
    Args:
        df: DataFrame with price data
        price_col: Column name for the price
        lookback: Number of historical hourly prices to use (168 = 1 week)
        forecast_horizon: Number of hours ahead to forecast (24 = 1 day)
    """
    df = df.copy()
    
    # Create lagged features
    for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
        df[f'price_lag_{lag}h'] = df[price_col].shift(lag)
    
    # Create rolling statistics
    for window in [24, 168]:
        df[f'price_mean_{window}h'] = df[price_col].rolling(window).mean().shift(1)
        df[f'price_std_{window}h'] = df[price_col].rolling(window).std().shift(1)
        df[f'price_max_{window}h'] = df[price_col].rolling(window).max().shift(1)
        df[f'price_min_{window}h'] = df[price_col].rolling(window).min().shift(1)
    
    # Create targets (prices for next 24 hours)
    for i in range(1, forecast_horizon + 1):
        df[f'target_{i}h'] = df[price_col].shift(-i)
    
    # Drop rows with NaNs
    df = df.dropna()
    
    # Separate features and target
    feature_columns = [col for col in df.columns if not col.startswith('target_') and col != price_col]
    target_columns = [f'target_{i}h' for i in range(1, forecast_horizon + 1)]
    
    # Scale features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
    
    # Scale targets
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    df[target_columns] = target_scaler.fit_transform(df[target_columns])
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - lookback + 1):
        X.append(df[feature_columns].iloc[i:i+lookback].values)
        y.append(df[target_columns].iloc[i+lookback-1].values)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, feature_scaler, target_scaler, df

# 3. Model Building
def build_lstm_model(lookback, n_features, n_outputs=24):
    """
    Build LSTM model for time series forecasting
    """
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(lookback, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs))

    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Custom callback to track original MSE and MAE
class OriginalMetricsCallback(Callback):
    def __init__(self, target_scaler, X_train, y_train, X_val, y_val):
        super(OriginalMetricsCallback, self).__init__()
        self.target_scaler = target_scaler
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_original_mse = []
        self.train_original_mae = []
        self.val_original_mse = []
        self.val_original_mae = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for validation data
        y_val_pred = self.model.predict(self.X_val, verbose=0)
        
        # Inverse transform validation predictions and actual values
        y_val_pred_original = self.target_scaler.inverse_transform(y_val_pred)
        y_val_original = self.target_scaler.inverse_transform(self.y_val)
        
        # Calculate original MSE and MAE for validation data
        val_mse = np.mean(np.square(y_val_original - y_val_pred_original))
        val_mae = np.mean(np.abs(y_val_original - y_val_pred_original))
        
        # Store validation metrics
        self.val_original_mse.append(val_mse)
        self.val_original_mae.append(val_mae)
        
        # Get predictions for training data
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        
        # Inverse transform training predictions and actual values
        y_train_pred_original = self.target_scaler.inverse_transform(y_train_pred)
        y_train_original = self.target_scaler.inverse_transform(self.y_train)
        
        # Calculate original MSE and MAE for training data
        train_mse = np.mean(np.square(y_train_original - y_train_pred_original))
        train_mae = np.mean(np.abs(y_train_original - y_train_pred_original))
        
        # Store training metrics
        self.train_original_mse.append(train_mse)
        self.train_original_mae.append(train_mae)
        
        # Add to logs
        logs['train_original_mse'] = train_mse
        logs['train_original_mae'] = train_mae
        logs['val_original_mse'] = val_mse
        logs['val_original_mae'] = val_mae

# Learning rate scheduler function
def lr_schedule(epoch, initial_lr=0.01, decay_factor=0.5, decay_epochs=10):
    """
    Learning rate scheduler function that reduces the learning rate by a factor
    after a certain number of epochs.
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_factor: Factor by which to reduce the learning rate
        decay_epochs: Number of epochs after which to reduce the learning rate
        
    Returns:
        Learning rate for the current epoch
    """
    if epoch % decay_epochs == 0 and epoch > 0:
        return initial_lr * (decay_factor ** (epoch // decay_epochs))
    return initial_lr

# 4. Training and Evaluation
def train_and_evaluate_model(X, y, test_size=0.2, epochs=100, batch_size=32, target_scaler=None):
    """
    Train and evaluate LSTM model
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of data to use for testing
        epochs: Number of training epochs
        batch_size: Batch size for training
        target_scaler: Scaler used for target values (for original metrics calculation)
    """
    # Split data into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = build_lstm_model(X.shape[1], X.shape[2], y.shape[1])
    
    # Create validation split
    val_split_idx = int(len(X_train) * 0.8)
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    X_train_subset = X_train[:val_split_idx]
    y_train_subset = y_train[:val_split_idx]
    
    # Create custom callback for original metrics if target_scaler is provided
    callbacks = []
    
    # Early stopping - use val_original_mae if target_scaler is provided, otherwise use val_loss
    if target_scaler is not None:
        # Create the original metrics callback first
        original_metrics_callback = OriginalMetricsCallback(
            target_scaler, 
            X_train_subset, 
            y_train_subset, 
            X_val, 
            y_val
        )
        callbacks.append(original_metrics_callback)
        
        # Early stopping on original MAE
        early_stopping = EarlyStopping(
            monitor='val_original_mae',
            patience=15,
            restore_best_weights=True,
            mode='min'
        )
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_original_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
    else:
        # Early stopping on validation loss
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    
    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    
    # Add callbacks
    callbacks.append(early_stopping)
    callbacks.append(reduce_lr)
    callbacks.append(lr_scheduler)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate original metrics if target_scaler is provided
    if target_scaler is not None:
        # Inverse transform predictions and actual values
        y_train_pred_original = target_scaler.inverse_transform(y_train_pred)
        y_train_original = target_scaler.inverse_transform(y_train)
        y_test_pred_original = target_scaler.inverse_transform(y_test_pred)
        y_test_original = target_scaler.inverse_transform(y_test)
        
        # Calculate original MSE and MAE
        train_mse_original = np.mean(np.square(y_train_original - y_train_pred_original))
        train_mae_original = np.mean(np.abs(y_train_original - y_train_pred_original))
        test_mse_original = np.mean(np.square(y_test_original - y_test_pred_original))
        test_mae_original = np.mean(np.abs(y_test_original - y_test_pred_original))
        
        print(f'Train Original MSE: {train_mse_original:.4f}, Train Original MAE: {train_mae_original:.4f}')
        print(f'Test Original MSE: {test_mse_original:.4f}, Test Original MAE: {test_mae_original:.4f}')
        
        # Plot original metrics during training
        plt.figure(figsize=(12, 10))
        
        # Plot original MSE
        plt.subplot(2, 2, 1)
        plt.plot(original_metrics_callback.train_original_mse, label='Training Original MSE')
        plt.plot(original_metrics_callback.val_original_mse, label='Validation Original MSE')
        plt.title('Original MSE During Training')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot original MAE
        plt.subplot(2, 2, 2)
        plt.plot(original_metrics_callback.train_original_mae, label='Training Original MAE')
        plt.plot(original_metrics_callback.val_original_mae, label='Validation Original MAE')
        plt.title('Original MAE During Training')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save the original metrics plot
        import os
        os.makedirs('./plots/', exist_ok=True)
        plt.savefig('./plots/original_training_metrics.png', dpi=300)
        plt.close()
    
    return model, history, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

# 5. Visualization
def visualize_results(df, y_test, y_test_pred, target_scaler, test_dates, price_col='SpotPriceDKK', save_path='./plots/'):
    """
    Visualize prediction results and save the figures
    """
    # Create directory for plots if it doesn't exist
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Inverse transform predictions
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_test_pred)
    
    # Calculate metrics for each forecast hour
    mse = []
    mae = []
    for i in range(y_test_inv.shape[1]):
        mse.append(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i]))
        mae.append(mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i]))
    
    # Plot metrics by forecast hour
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 25), mse, 'o-', label='MSE')
    plt.title('MSE by Forecast Hour')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 25), mae, 'o-', label='MAE')
    plt.title('MAE by Forecast Hour')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the metrics plot
    plt.savefig(os.path.join(save_path, 'forecast_metrics_by_hour.png'), dpi=300)
    plt.close()
    
    # Plot example forecasts
    # Select a random sample of 3 days
    n_days = 3
    sample_indices = np.random.choice(range(len(y_test_inv)), n_days, replace=False)
    
    plt.figure(figsize=(15, 5 * n_days))
    for i, idx in enumerate(sample_indices):
        actual_prices = y_test_inv[idx]
        predicted_prices = y_pred_inv[idx]
        
        # Get the date for this forecast
        forecast_date = test_dates[idx]
        
        # Create hour timestamps
        forecast_hours = [forecast_date + timedelta(hours=h) for h in range(1, 25)]
        
        plt.subplot(n_days, 1, i+1)
        plt.plot(forecast_hours, actual_prices, 'b-', label='Actual Price')
        plt.plot(forecast_hours, predicted_prices, 'r--', label='Predicted Price')
        plt.title(f'Day-Ahead Forecast for {forecast_date.date()}')
        plt.xlabel('Hour')
        plt.ylabel(f'{price_col}')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the sample forecasts plot
    plt.savefig(os.path.join(save_path, 'sample_forecasts.png'), dpi=300)
    plt.close()
    
    # Calculate overall metrics
    overall_mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
    overall_mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
    overall_r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
    
    print(f'Overall MSE: {overall_mse:.2f}')
    print(f'Overall MAE: {overall_mae:.2f}')
    print(f'Overall R²: {overall_r2:.4f}')
    
    return mse, mae, overall_mse, overall_mae, overall_r2

# 6. Forecast Generation for Next Day
def generate_day_ahead_forecast(model, latest_data, feature_scaler, target_scaler, last_known_timestamp, price_col='SpotPriceDKK'):
    """
    Generate forecasts for the next 24 hours
    
    Parameters:
    model: Trained LSTM model
    latest_data: NumPy array containing the latest feature data
    feature_scaler: Scaler used for features
    target_scaler: Scaler used for targets
    last_known_timestamp: The timestamp of the last known data point
    price_col: Column name for price data
    """
    # Prepare the input data (already scaled)
    input_data = latest_data.reshape(1, latest_data.shape[0], latest_data.shape[1])
    
    # Generate prediction
    prediction_scaled = model.predict(input_data)
    
    # Inverse transform
    prediction = target_scaler.inverse_transform(prediction_scaled)[0]
    
    # Create timestamp for the next 24 hours
    forecast_hours = [last_known_timestamp + timedelta(hours=i+1) for i in range(24)]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'HourUTC': forecast_hours,
        f'Forecast_{price_col}': prediction
    })
    forecast_df.set_index('HourUTC', inplace=True)
    
    return forecast_df

# 7. Full Pipeline
def run_electricity_price_forecasting(file_path, price_col='SpotPriceDKK', save_path='./plots/'):
    """
    Run the full electricity price forecasting pipeline
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    # 2. Prepare data for LSTM
    print("Preparing data for LSTM...")
    X, y, feature_scaler, target_scaler, processed_df = prepare_lstm_data(df, price_col=price_col)
    
    # 3. Train and evaluate model
    print("Training and evaluating model...")
    model, history, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_and_evaluate_model(X, y, target_scaler=target_scaler)
    
    # Get test dates
    split_idx = int(len(processed_df) * 0.8)  # Same split as in train_and_evaluate
    test_dates = processed_df.index[split_idx:split_idx+len(y_test)]
    
    # 4. Visualize results
    print("Visualizing results...")
    mse, mae, overall_mse, overall_mae, overall_r2 = visualize_results(
        df, y_test, y_test_pred, target_scaler, test_dates, price_col, save_path
    )
    
    # 5. Generate forecast for next day (using the last available data)
    print("Generating forecast for next day...")
    latest_data = X_test[-1]  # Use the last test sample
    
    # Get the last known timestamp from the processed dataframe
    last_known_timestamp = processed_df.index[len(processed_df)-1]
    
    forecast_df = generate_day_ahead_forecast(
        model, latest_data, feature_scaler, target_scaler, last_known_timestamp, price_col
    )
    
    print("\nDay-ahead forecast:")
    print(forecast_df)
    
    # Plot forecast
    import os
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df.index, forecast_df[f'Forecast_{price_col}'], 'r-o', label='Forecast')
    plt.title('Day-Ahead Electricity Price Forecast')
    plt.xlabel('Hour')
    plt.ylabel(price_col)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the day-ahead forecast plot
    plt.savefig(os.path.join(save_path, 'day_ahead_forecast.png'), dpi=300)
    plt.close()
    
    # Also save the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300)
    plt.close()
    
    return model, feature_scaler, target_scaler, forecast_df, overall_mse, overall_mae, overall_r2

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "data/electricity_hourly_price_5y.csv"
    model, feature_scaler, target_scaler, forecast_df, mse, mae, r2 = run_electricity_price_forecasting(file_path)
    
    # Save model and scalers for future use
    model.save("electricity_price_model.h5")
    
    print(f"Model and forecasting pipeline completed successfully.")
    print(f"Overall MSE: {mse:.2f}")
    print(f"Overall MAE: {mae:.2f}")
    print(f"Overall R²: {r2:.4f}")