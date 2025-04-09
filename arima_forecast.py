import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the electricity price data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert HourUTC to datetime and set as index
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])
    df = df.set_index('HourUTC')
    
    # Sort by datetime
    df = df.sort_index()
    
    # Handle missing values
    if df['SpotPriceDKK'].isnull().sum() > 0:
        df['SpotPriceDKK'] = df['SpotPriceDKK'].interpolate(method='time')
    
    return df

def check_stationarity(series):
    """
    Check if the time series is stationary using Augmented Dickey-Fuller test
    """
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] <= 0.05:
        print("The series is stationary")
        return True
    else:
        print("The series is not stationary")
        return False

def make_stationary(series):
    """
    Make the time series stationary using differencing
    """
    # First difference
    diff1 = series.diff().dropna()
    
    # Check if stationary
    if check_stationarity(diff1):
        return diff1, 1
    
    # Second difference if needed
    diff2 = diff1.diff().dropna()
    if check_stationarity(diff2):
        return diff2, 2
    
    return series, 0

def find_optimal_arima_params(series):
    """
    Find optimal ARIMA parameters using ACF and PACF plots
    """
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax1, lags=50)
    plot_pacf(series, ax=ax2, lags=50)
    plt.tight_layout()
    plt.savefig('acf_pacf_plots.png')
    plt.close()
    
    # Based on ACF and PACF plots, suggest initial parameters
    # These can be adjusted based on the plots
    p = 2  # AR order
    d = 1  # Differencing order
    q = 2  # MA order
    
    return p, d, q

def train_arima_model(series, p, d, q):
    """
    Train ARIMA model with given parameters
    """
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def forecast_future(model_fit, steps=30):
    """
    Generate future forecasts
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

def plot_forecast(original_series, forecast, save_path='forecast_plot.png'):
    """
    Plot the last 24 hours of historical data and the 24-hour forecast
    """
    # Get the last 24 hours of historical data
    last_24_hours = original_series.iloc[-24:]  # Last 24 hours
    
    plt.figure(figsize=(12, 6))
    plt.plot(last_24_hours.index, last_24_hours.values, label='Last 24 Hours Historical')
    plt.plot(pd.date_range(start=last_24_hours.index[-1], periods=len(forecast)+1, freq='H')[1:], 
             forecast, label='24-Hour Forecast', color='red')
    plt.title('Electricity Price: Last 24 Hours and 24-Hour Forecast')
    plt.xlabel('Time')
    plt.ylabel('Spot Price (DKK)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data/electricity_hourly_price_5y.csv')
    
    # Extract the price series
    price_series = df['SpotPriceDKK']
    
    # Check stationarity
    print("\nChecking stationarity...")
    is_stationary = check_stationarity(price_series)
    
    # Make series stationary if needed
    if not is_stationary:
        print("\nMaking series stationary...")
        stationary_series, d = make_stationary(price_series)
    else:
        stationary_series = price_series
        d = 0
    
    # Find optimal ARIMA parameters
    print("\nFinding optimal ARIMA parameters...")
    p, d, q = find_optimal_arima_params(stationary_series)
    print(f"Suggested ARIMA parameters: p={p}, d={d}, q={q}")
    
    # Train ARIMA model
    print("\nTraining ARIMA model...")
    model_fit = train_arima_model(price_series, p, d, q)
    
    # Generate forecast
    print("\nGenerating forecast...")
    forecast = forecast_future(model_fit, steps=24)  # 24 hours forecast
    
    # Plot results
    print("\nPlotting results...")
    plot_forecast(price_series, forecast)
    
    print("\nForecast completed successfully!")
    print("Results have been saved to 'forecast_plot.png'")

if __name__ == "__main__":
    main() 