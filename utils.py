import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from sklearn.model_selection import TimeSeriesSplit

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional time-based features
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['HourUTC'].dt.hour
    df['day'] = df['HourUTC'].dt.day
    df['month'] = df['HourUTC'].dt.month
    df['year'] = df['HourUTC'].dt.year
    df['dayofweek'] = df['HourUTC'].dt.dayofweek
    df['quarter'] = df['HourUTC'].dt.quarter
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       lag_hours: List[int] = [1, 2, 3, 24, 48, 168]) -> pd.DataFrame:
    """
    Create lagged features for the target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with target column
    target_col : str
        Name of the target column
    lag_hours : List[int]
        List of lag hours to create
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional lagged features
    """
    df = df.copy()
    
    for lag in lag_hours:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
    return df

def create_rolling_features(df: pd.DataFrame,
                          target_col: str,
                          windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Create rolling statistics features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with target column
    target_col : str
        Name of the target column
    windows : List[int]
        List of window sizes for rolling statistics
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        
    return df

def prepare_data(df: pd.DataFrame,
                target_col: str = 'SpotPriceDKK',
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
        df['year_utc'] = df['HourUTC'].dt.year
        df['dayofweek_utc'] = df['HourUTC'].dt.dayofweek
        df['quarter_utc'] = df['HourUTC'].dt.quarter
        df.drop('HourUTC', axis=1, inplace=True)
    
    if 'HourDK' in df.columns:
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        df['hour_dk'] = df['HourDK'].dt.hour
        df['day_dk'] = df['HourDK'].dt.day
        df['month_dk'] = df['HourDK'].dt.month
        df['year_dk'] = df['HourDK'].dt.year
        df['dayofweek_dk'] = df['HourDK'].dt.dayofweek
        df['quarter_dk'] = df['HourDK'].dt.quarter
        df.drop('HourDK', axis=1, inplace=True)
    
    # Convert categorical columns to numerical using label encoding
    if 'PriceArea' in df.columns:
        df['PriceArea'] = df['PriceArea'].astype('category').cat.codes
    
    # Identify feature columns (excluding target)
    feature_cols = [col for col in df.columns if col != target_col]
    
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

def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    # Identify numerical columns (excluding categorical columns)
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create scaler for numerical features
    scaler = StandardScaler()
    
    # Scale numerical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Only scale numerical columns
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train_scaled, X_test_scaled 