import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from data_preparation import load_hourly_data_split

def load_model(model_path: str):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    
    Returns:
    --------
    object
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def predict_with_model(model: object,
                      X: pd.DataFrame,
                      model_type: str = 'xgboost') -> np.ndarray:
    """
    Make predictions using a trained model
    
    Parameters:
    -----------
    model : object
        Trained model
    X : pd.DataFrame
        Features for prediction
    model_type : str
        Type of model ('xgboost' or 'lightgbm')
    
    Returns:
    --------
    np.ndarray
        Predicted values
    """
    return model.predict(X)

def evaluate_predictions(y_true: pd.Series,
                        y_pred: np.ndarray,
                        hour: int,
                        model_type: str,
                        save_dir: str = 'results') -> Dict[str, float]:
    """
    Evaluate predictions and save results
    
    Parameters:
    -----------
    y_true : pd.Series
        Actual values
    y_pred : np.ndarray
        Predicted values
    hour : int
        Hour of the day
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{save_dir}/metrics_hour_{hour}_{model_type}.csv', index=False)
    
    # Create and save prediction plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted')
    plt.title(f'Predictions vs Actual Values (Hour {hour})')
    plt.xlabel('Time')
    plt.ylabel('Price (DKK)')
    plt.legend()
    plt.savefig(f'{save_dir}/predictions_hour_{hour}_{model_type}.png')
    plt.close()
    
    return metrics

def predict_with_hourly_models(data_dir: str = 'data',
                             models_dir: str = 'models',
                             results_dir: str = 'results') -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Make predictions using hourly models
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the hourly data splits
    models_dir : str
        Directory containing the trained models
    results_dir : str
        Directory to save results
    
    Returns:
    --------
    Dict[str, Dict[int, Dict[str, float]]]
        Dictionary mapping model types to their hourly metrics
    """
    # Load hourly data
    hourly_data = load_hourly_data_split(data_dir)
    
    # Initialize results dictionary
    all_metrics = {}
    
    # Make predictions for each model type
    for model_type in ['xgboost', 'lightgbm']:
        print(f"\nMaking predictions with {model_type} models...")
        model_metrics = {}
        
        for hour, (X_train, X_test, y_train, y_test) in hourly_data.items():
            print(f"\nProcessing hour {hour}")
            
            try:
                # Load model
                model_path = f'{models_dir}/hour_{hour}/{model_type}_model.joblib'
                model = load_model(model_path)
                
                # Make predictions
                y_pred = predict_with_model(model, X_test, model_type)
                
                # Evaluate predictions
                metrics = evaluate_predictions(y_test, y_pred, hour, model_type, results_dir)
                model_metrics[hour] = metrics
                
                print(f"Metrics for hour {hour}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
            except Exception as e:
                print(f"Error processing hour {hour}: {str(e)}")
                model_metrics[hour] = None
        
        all_metrics[model_type] = model_metrics
    
    return all_metrics

if __name__ == "__main__":
    # Make predictions with all hourly models
    all_metrics = predict_with_hourly_models()
    
    # Print overall results
    print("\nOverall Prediction Metrics:")
    for model_type, hourly_metrics in all_metrics.items():
        print(f"\n{model_type.upper()}:")
        for hour in range(24):
            if hourly_metrics[hour] is not None:
                print(f"\nHour {hour}:")
                for metric_name, value in hourly_metrics[hour].items():
                    print(f"{metric_name}: {value:.4f}") 