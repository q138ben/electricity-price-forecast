import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import joblib
import os

def prepare_data(df: pd.DataFrame,
                target_col: str = 'target_price',
                test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for modeling by splitting features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features and target
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Training features, test features, training target, test target
    """
    df = df.copy()
    
    # Convert datetime columns to numerical features
    if 'HourUTC' in df.columns:
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        df['hour_utc'] = df['HourUTC'].dt.hour
        df['day_utc'] = df['HourUTC'].dt.day
        df['month_utc'] = df['HourUTC'].dt.month
        df['dayofweek_utc'] = df['HourUTC'].dt.dayofweek
        df['quarter_utc'] = df['HourUTC'].dt.quarter
        df.drop('HourUTC', axis=1, inplace=True)
    
    df.drop('PriceArea', axis=1, inplace=True)
    df.drop('HourDK', axis=1, inplace=True)
    df.drop('SpotPriceEUR', axis=1, inplace=True)
 
    # Create the target: price 24 hours ahead
    df[target_col] = df['SpotPriceDKK'].shift(-24)
    
    # Lag features (all these will be available at prediction time)
    for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
        df[f'price_lag_{lag}h'] = df['SpotPriceDKK'].shift(lag)
    
    df.dropna(inplace=True)
    
    # Identify feature columns (excluding target)
    feature_cols = [col for col in df.columns if col not in [target_col, 'SpotPriceDKK']]

    # Split features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    
    # Use the last split for train/test
    train_idx, test_idx = splits[-1]
    
    # Split the data
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test

def prepare_hourly_data(df: pd.DataFrame,
                       target_col: str = 'target_price',
                       test_size: float = 0.2) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Prepare data for modeling by splitting features and target for each hour
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features and target
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]
        Dictionary mapping each hour to its training and test data
    """
    df = df.copy()
    
    # Convert datetime columns to numerical features
    if 'HourUTC' in df.columns:
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        df['hour_utc'] = df['HourUTC'].dt.hour
        df['day_utc'] = df['HourUTC'].dt.day
        df['month_utc'] = df['HourUTC'].dt.month
        df['dayofweek_utc'] = df['HourUTC'].dt.dayofweek
        df['quarter_utc'] = df['HourUTC'].dt.quarter
        df['year_utc'] = df['HourUTC'].dt.year
        df['is_weekend'] = df['dayofweek_utc'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['dayofweek_utc'].isin([5, 6]).astype(int)  # Simplified holiday detection
        df.drop('HourUTC', axis=1, inplace=True)
    
    df.drop('PriceArea', axis=1, inplace=True)
    df.drop('HourDK', axis=1, inplace=True)
    df.drop('SpotPriceEUR', axis=1, inplace=True)
 
    # Create the target: price 24 hours ahead
    df[target_col] = df['SpotPriceDKK'].shift(-24)
    
    # Lag features (all these will be available at prediction time)
    for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
        df[f'price_lag_{lag}h'] = df['SpotPriceDKK'].shift(lag)
        # Add rolling statistics
        df[f'price_rolling_mean_{lag}h'] = df['SpotPriceDKK'].rolling(window=lag).mean()
        df[f'price_rolling_std_{lag}h'] = df['SpotPriceDKK'].rolling(window=lag).std()
        df[f'price_rolling_max_{lag}h'] = df['SpotPriceDKK'].rolling(window=lag).max()
        df[f'price_rolling_min_{lag}h'] = df['SpotPriceDKK'].rolling(window=lag).min()
    
    # Add price differences
    for lag in [24, 48, 72, 168]:
        df[f'price_diff_{lag}h'] = df['SpotPriceDKK'] - df[f'price_lag_{lag}h']
    
    # Add seasonal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_utc'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_utc'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month_utc'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_utc'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek_utc'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek_utc'] / 7)
    
    # Add price momentum features
    for window in [3, 6, 12, 24]:
        df[f'price_momentum_{window}h'] = df['SpotPriceDKK'].pct_change(window)
    
    # Add price volatility features
    for window in [3, 6, 12, 24]:
        df[f'price_volatility_{window}h'] = df['SpotPriceDKK'].rolling(window=window).std()
    
    df.dropna(inplace=True)
    
    # Identify feature columns (excluding target)
    feature_cols = [col for col in df.columns if col not in [target_col, 'SpotPriceDKK']]

    # Split features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    
    # Use the last split for train/test
    train_idx, test_idx = splits[-1]
    
    # Split the data
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # Create hourly splits
    hourly_data = {}
    for hour in range(24):
        # Get data for this hour
        train_mask = X_train['hour_utc'] == hour
        test_mask = X_test['hour_utc'] == hour
        
        hourly_data[hour] = (
            X_train[train_mask],
            X_test[test_mask],
            y_train[train_mask],
            y_test[test_mask]
        )
    
    return hourly_data

def scale_features(X_train: pd.DataFrame,
                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Scaled training and test features
    """
    # Identify numerical columns (excluding categorical columns)
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create scaler for numerical features
    scaler = StandardScaler()
    
    # Scale numerical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train_scaled, X_test_scaled

def save_data_split(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   save_dir: str = 'data') -> None:
    """
    Save the data split to files
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
    save_dir : str
        Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training data
    X_train.to_csv(f'{save_dir}/X_train.csv')
    y_train.to_csv(f'{save_dir}/y_train.csv')
    
    # Save test data
    X_test.to_csv(f'{save_dir}/X_test.csv')
    y_test.to_csv(f'{save_dir}/y_test.csv')
    
    print(f"Data split saved to '{save_dir}' directory")

def load_data_split(save_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the data split from files
    
    Parameters:
    -----------
    save_dir : str
        Directory containing the saved data
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Training features, test features, training target, test target
    """
    # Load training data
    X_train = pd.read_csv(f'{save_dir}/X_train.csv', index_col=0)
    y_train = pd.read_csv(f'{save_dir}/y_train.csv', index_col=0)
    
    # Load test data
    X_test = pd.read_csv(f'{save_dir}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{save_dir}/y_test.csv', index_col=0)
    
    return X_train, X_test, y_train, y_test

def save_hourly_data_split(hourly_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
                         save_dir: str = 'data') -> None:
    """
    Save the hourly data splits to files
    
    Parameters:
    -----------
    hourly_data : Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]
        Dictionary mapping each hour to its training and test data
    save_dir : str
        Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
        # Create hour-specific directory
        hour_dir = f'{save_dir}/hour_{hour}'
        os.makedirs(hour_dir, exist_ok=True)
        
        # Save training data
        X_train.to_csv(f'{hour_dir}/X_train.csv')
        y_train.to_csv(f'{hour_dir}/y_train.csv')
        
        # Save test data
        X_test.to_csv(f'{hour_dir}/X_test.csv')
        y_test.to_csv(f'{hour_dir}/y_test.csv')
    
    print(f"Hourly data splits saved to '{save_dir}' directory")

def load_hourly_data_split(save_dir: str = 'data') -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Load the hourly data splits from files
    
    Parameters:
    -----------
    save_dir : str
        Directory containing the saved data
    
    Returns:
    --------
    Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]
        Dictionary mapping each hour to its training and test data
    """
    hourly_data = {}
    
    for hour in range(24):
        hour_dir = f'{save_dir}/hour_{hour}'
        
        # Load training data
        X_train = pd.read_csv(f'{hour_dir}/X_train.csv', index_col=0)
        y_train = pd.read_csv(f'{hour_dir}/y_train.csv', index_col=0)
        
        # Load test data
        X_test = pd.read_csv(f'{hour_dir}/X_test.csv', index_col=0)
        y_test = pd.read_csv(f'{hour_dir}/y_test.csv', index_col=0)
        
        hourly_data[hour] = (X_train, X_test, y_train, y_test)
    
    return hourly_data

if __name__ == "__main__":
    # Example usage
    from data_fetcher import EnerginetDataFetcher
    
    # Fetch data
    fetcher = EnerginetDataFetcher()
    df = fetcher.fetch_historical_data(days=730)
    
    # Prepare and split data for each hour
    hourly_data = prepare_hourly_data(df)
    
    # Scale features for each hour
    hourly_scaled_data = {}
    for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        hourly_scaled_data[hour] = (X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save hourly data splits
    save_hourly_data_split(hourly_scaled_data) 