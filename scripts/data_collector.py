"""
Stock Data Collection Module
CS 5800 - Algorithms Final Project
Authors: Dev Patel, Kunal Ghanwat

This module handles data acquisition from Yahoo Finance API.
Time Complexity: O(n) where n is the number of trading days
Space Complexity: O(n*m) where m is the number of stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Collects and manages historical stock data.
    CS 5800 Reference: Implements efficient data structures (Module 3)
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize data collector with stock tickers and date range.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}  # Hash table for O(1) lookup - CS 5800 Module 3
        
    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical data for a single stock.
        
        Time Complexity: O(n) where n is number of trading days
        Space Complexity: O(n)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {ticker}")
            
            # Download data with error handling
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Add ticker column for identification
            df['Ticker'] = ticker
            
            # Calculate additional metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def fetch_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all stocks in parallel.
        
        Time Complexity: O(n*m) where n is days and m is stocks
        Space Complexity: O(n*m)
        """
        all_data = {}
        
        for ticker in self.tickers:
            try:
                df = self.fetch_stock_data(ticker)
                all_data[ticker] = df
                self.data_cache[ticker] = df  # Cache for quick access
            except Exception as e:
                logger.warning(f"Skipping {ticker} due to error: {e}")
                continue
        
        return all_data
    
    def save_to_csv(self, output_dir: str = "data/raw"):
        """
        Save collected data to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for ticker, df in self.data_cache.items():
            filepath = os.path.join(output_dir, f"{ticker}_historical.csv")
            df.to_csv(filepath)
            logger.info(f"Saved {ticker} data to {filepath}")
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """
        Combine all stock data into single DataFrame.
        
        CS 5800 Reference: Merge operation similar to MergeSort (Module 2)
        """
        if not self.data_cache:
            self.fetch_all_stocks()
        
        combined_frames = []
        for ticker, df in self.data_cache.items():
            combined_frames.append(df)
        
        return pd.concat(combined_frames, axis=0)

# Main execution
if __name__ == "__main__":
    # Define parameters
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
    START_DATE = '2020-01-01'
    END_DATE = '2025-11-01'
    
    # Initialize collector
    collector = StockDataCollector(TICKERS, START_DATE, END_DATE)
    
    # Fetch all data
    stock_data = collector.fetch_all_stocks()
    
    # Save to CSV
    collector.save_to_csv()
    
    # Print summary statistics
    for ticker, df in stock_data.items():
        print(f"\n{ticker} Statistics:")
        print(f"Total trading days: {len(df)}")
        print(f"Average daily return: {df['Daily_Return'].mean():.4%}")
        print(f"Volatility (std): {df['Daily_Return'].std():.4%}")
        print(f"Max price: ${df['Close'].max():.2f}")
        print(f"Min price: ${df['Close'].min():.2f}")