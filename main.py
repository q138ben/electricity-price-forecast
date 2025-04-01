import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from datetime import datetime, timedelta

from data_fetcher import EnerginetDataFetcher
from utils import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    prepare_data,
    scale_features
)

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
sns.set_palette('husl')

def fetch_and_prepare_data():
    """Fetch and prepare data for analysis"""
    print("Fetching data from Energinet...")
    fetcher = EnerginetDataFetcher()
    df = fetcher.fetch_historical_data(days=730)  # Last 2 years of data
    
    print("\nDataset Info:")
    print(df.info())
    print("\nSample of the data:")
    print(df.head())
    
    return df

def create_features(df):
    """Create features for modeling"""
    print("\nCreating features...")
    
    # Create time-based features
    df = create_time_features(df)
    
    # Create lag features
    df = create_lag_features(df, 'SpotPriceDKK')
    
    # Create rolling features
    df = create_rolling_features(df, 'SpotPriceDKK')
    
    print("\nNew features created:")
    print(df.columns.tolist())
    
    return df

def exploratory_data_analysis(df):
    """Perform exploratory data analysis"""
    print("\nPerforming exploratory data analysis...")
    
    # Plot price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='SpotPriceDKK', bins=50)
    plt.title('Distribution of Electricity Prices')
    plt.xlabel('Price (DKK)')
    plt.ylabel('Count')
    plt.savefig('price_distribution.png')
    plt.close()
    
    # Plot price over time
    plt.figure(figsize=(15, 6))
    plt.plot(df['HourUTC'], df['SpotPriceDKK'])
    plt.title('Electricity Prices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price (DKK)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_over_time.png')
    plt.close()
    
    # Plot average price by hour of day
    hourly_avg = df.groupby('hour')['SpotPriceDKK'].mean()
    plt.figure(figsize=(12, 6))
    hourly_avg.plot(kind='bar')
    plt.title('Average Price by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Price (DKK)')
    plt.tight_layout()
    plt.savefig('hourly_average.png')
    plt.close()

def train_short_term_models(df):
    """Train and evaluate short-term forecasting models"""
    print("\nTraining short-term forecasting models...")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Initialize models
    models = {
        'XGBoost': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'LightGBM': lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} Results:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")
    
    return results, models, X_test_scaled, y_test

def train_long_term_model(df):
    """Train and evaluate long-term forecasting model using Prophet"""
    print("\nTraining long-term forecasting model (Prophet)...")
    
    # Prepare data for Prophet
    prophet_df = df[['HourUTC', 'SpotPriceDKK']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Initialize and fit Prophet model
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    prophet_model.fit(prophet_df)
    
    # Create future dates for forecasting
    future_dates = prophet_model.make_future_dataframe(periods=24*7)  # Forecast next week
    
    # Make predictions
    forecast = prophet_model.predict(future_dates)
    
    # Plot the forecast
    plt.figure(figsize=(15, 8))
    prophet_model.plot(forecast)
    plt.title('Prophet Forecast for Next Week')
    plt.xlabel('Date')
    plt.ylabel('Price (DKK)')
    plt.tight_layout()
    plt.savefig('prophet_forecast.png')
    plt.close()
    
    return prophet_model, forecast

def plot_model_comparison(results, models, X_test_scaled, y_test):
    """Plot comparison of model predictions"""
    print("\nPlotting model comparison...")
    
    # Compare model performance
    results_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(results_df)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        plt.plot(y_test.index, y_pred, label=f'{name} Predictions', alpha=0.7)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price (DKK)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    """Main function to run the analysis"""
    # Fetch and prepare data
    df = fetch_and_prepare_data()
    
    # Create features
    df = create_features(df)
    
    # Perform exploratory data analysis
    exploratory_data_analysis(df)
    
    # Train and evaluate short-term models
    results, models, X_test_scaled, y_test = train_short_term_models(df)
    
    # Train and evaluate long-term model
    prophet_model, forecast = train_long_term_model(df)
    
    # Plot model comparison
    plot_model_comparison(results, models, X_test_scaled, y_test)
    
    print("\nAnalysis complete! Check the generated plots for results.")

if __name__ == "__main__":
    main() 