# Electricity Price Forecasting

This project implements a solution for short- and long-term electricity price forecasting using data from Energinet's open data platform. The solution uses machine learning models to predict day-ahead electricity prices in Denmark.

## Project Structure

```
electricity-price-forecast/
├── data_fetcher.py      # Script for fetching data from Energinet API
├── utils.py            # Utility functions for data processing
├── electricity_price_forecast.ipynb  # Main analysis notebook
├── environment.yml     # Conda environment configuration
├── requirements.txt    # Project dependencies (alternative)
└── README.md          # Project documentation
```

## Features

- Data collection from Energinet's open data platform
- Feature engineering including:
  - Time-based features
  - Lagged features
  - Rolling statistics
- Multiple modeling approaches:
  - XGBoost for short-term forecasting
  - LightGBM for short-term forecasting
  - Prophet for long-term forecasting
- Comprehensive evaluation metrics
- Visualization of results and forecasts

## Setup

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/electricity-price-forecast.git
cd electricity-price-forecast
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate electricity-price-forecast
```

3. Launch JupyterLab:
```bash
jupyter lab
```

4. Open `electricity_price_forecast.ipynb` to run the analysis

### Alternative: Using pip and venv

If you prefer using pip and venv instead of conda:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/electricity-price-forecast.git
cd electricity-price-forecast
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch JupyterLab:
```bash
jupyter lab
```

5. Open `electricity_price_forecast.ipynb` to run the analysis

## Usage

The main analysis is contained in the Jupyter notebook `electricity_price_forecast.ipynb`. The notebook includes:

1. Data collection and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model development and evaluation
5. Long-term forecasting
6. Results analysis and visualization

## Assumptions and Limitations

1. Data Quality:
   - Assumes reliable and consistent data from Energinet
   - Handles missing values and outliers

2. Model Assumptions:
   - Stationarity in price patterns
   - Historical patterns continue into the future
   - No major market disruptions

3. Limitations:
   - Limited to historical data patterns
   - May not capture sudden market changes
   - External factors not included (weather, political events, etc.)

## Future Work

1. Feature Engineering:
   - Incorporate weather data
   - Add holiday and special event indicators
   - Consider cross-border electricity flows

2. Model Improvements:
   - Try ensemble methods
   - Implement deep learning models
   - Add uncertainty quantification

3. Additional Analysis:
   - Analyze price volatility patterns
   - Study the impact of renewable energy sources
   - Investigate market dynamics

## License

This project is licensed under the MIT License - see the LICENSE file for details.