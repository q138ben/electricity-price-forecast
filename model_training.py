import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import joblib
import os
from typing import Dict, Tuple, List
from data_preparation import load_hourly_data_split

def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  save_dir: str = 'models') -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train and evaluate XGBoost model
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    Tuple[xgb.XGBRegressor, Dict[str, float]]
        Trained model and evaluation metrics
    """
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, f'{save_dir}/xgboost_model.joblib')
    
    return model, metrics

def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   save_dir: str = 'models') -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """
    Train and evaluate LightGBM model
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    Tuple[lgb.LGBMRegressor, Dict[str, float]]
        Trained model and evaluation metrics
    """
    # Train model
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, f'{save_dir}/lightgbm_model.joblib')
    
    return model, metrics

def train_hourly_models(data_dir: str = 'data',
                       save_dir: str = 'models',
                       model_type: str = 'xgboost') -> Dict[int, Dict[str, float]]:
    """
    Train models for each hour of the day
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the hourly data splits
    save_dir : str
        Directory to save the models
    model_type : str
        Type of model to train ('xgboost' or 'lightgbm')
    
    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping each hour to its model metrics
    """
    # Load hourly data
    hourly_data = load_hourly_data_split(data_dir)
    
    # Initialize results dictionary
    hourly_metrics = {}
    
    # Train models for each hour
    for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
        print(f"\nTraining {model_type} model for hour {hour}")
        
        # Create hour-specific save directory
        hour_save_dir = f'{save_dir}/hour_{hour}'
        os.makedirs(hour_save_dir, exist_ok=True)
        
        try:
            # Train model based on type
            if model_type == 'xgboost':
                model, metrics = train_xgboost(X_train, y_train, X_test, y_test, hour_save_dir)
            elif model_type == 'lightgbm':
                model, metrics = train_lightgbm(X_train, y_train, X_test, y_test, hour_save_dir)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            hourly_metrics[hour] = metrics
            print(f"Metrics for hour {hour}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
        except Exception as e:
            print(f"Error training model for hour {hour}: {str(e)}")
            hourly_metrics[hour] = None
    
    return hourly_metrics

def train_all_hourly_models(data_dir: str = 'data',
                          save_dir: str = 'models') -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Train all types of models for each hour
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the hourly data splits
    save_dir : str
        Directory to save the models
    
    Returns:
    --------
    Dict[str, Dict[int, Dict[str, float]]]
        Dictionary mapping model types to their hourly metrics
    """
    all_metrics = {}
    
    # Train XGBoost models
    print("\nTraining XGBoost models...")
    all_metrics['xgboost'] = train_hourly_models(data_dir, save_dir, 'xgboost')
    
    # Train LightGBM models
    print("\nTraining LightGBM models...")
    all_metrics['lightgbm'] = train_hourly_models(data_dir, save_dir, 'lightgbm')
    
    return all_metrics

if __name__ == "__main__":
    # Train all hourly models
    all_metrics = train_all_hourly_models()
    
    # Print overall results
    print("\nOverall Model Performance Metrics:")
    for model_type, hourly_metrics in all_metrics.items():
        print(f"\n{model_type.upper()}:")
        for hour in range(24):
            if hourly_metrics[hour] is not None:
                print(f"\nHour {hour}:")
                for metric_name, value in hourly_metrics[hour].items():
                    print(f"{metric_name}: {value:.4f}") 