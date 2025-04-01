import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

class EnerginetDataFetcher:
    """Class to fetch electricity price data from Energinet's API"""
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset/Elspotprices"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_elspot_prices(self, start_date=None, end_date=None, price_area='DK1'):
        """
        Fetch electricity spot prices from Energinet's API
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            price_area (str): Price area (e.g., 'DK1', 'DK2')
        
        Returns:
            pd.DataFrame: DataFrame containing the spot prices
        """
        try:
            # Construct the query parameters
            params = {
                'limit': 100000,  # Maximum limit to get all data
                'filter': f'{{"PriceArea": "{price_area}"}}'
            }
            
            if start_date and end_date:
                params['start'] = start_date
                params['end'] = end_date
            
            # Make the API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse the response
            result = response.json()
            
            if not result.get('records'):
                self.logger.warning("No data found for the specified parameters")
                return pd.DataFrame()
            
            # Convert records to DataFrame
            df = pd.DataFrame(result['records'])
            
            # Convert timestamp to datetime
            df['HourUTC'] = pd.to_datetime(df['HourUTC'])
            
            # Sort by timestamp
            df = df.sort_values('HourUTC')
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, days=30, price_area='DK1'):
        """
        Fetch historical electricity spot prices
        
        Args:
            days (int): Number of days of historical data to fetch
            price_area (str): Price area (e.g., 'DK1', 'DK2')
        
        Returns:
            pd.DataFrame: DataFrame containing the historical spot prices
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_elspot_prices(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            price_area=price_area
        )

if __name__ == "__main__":
    # Example usage
    fetcher = EnerginetDataFetcher()
    df = fetcher.fetch_historical_data(days=100)
    print("\nLast 30 days of data:")
    print(df.head()) 