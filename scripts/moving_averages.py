"""
Moving Average Algorithms Implementation
CS 5800 - Algorithms Final Project

Implements SMA, EMA, and MACD with signal generation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MovingAverageAnalyzer:
    """
    Implements various moving average algorithms for trend analysis.
    CS 5800 Reference: Sliding window algorithms (Module 4)
    """
    
    def __init__(self):
        """Initialize the analyzer with default parameters."""
        self.sma_windows = [5, 10, 20, 50, 200]
        self.ema_spans = [12, 26, 50]
        self.signals_generated = False
        
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Time Complexity: O(n) using pandas rolling window
        Space Complexity: O(n) for output series
        
        CS 5800: Implements sliding window technique efficiently
        
        Args:
            prices: Series of prices
            window: Window size for averaging
            
        Returns:
            Series of SMA values
        """
        if len(prices) < window:
            return pd.Series(index=prices.index, dtype=float)
        
        # Using pandas rolling for O(n) complexity instead of naive O(n*w)
        sma = prices.rolling(window=window, min_periods=window).mean()
        return sma
    
    def calculate_ema(self, prices: pd.Series, span: int, 
                     adjust: bool = True) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Time Complexity: O(n) - single pass through data
        Space Complexity: O(n) for output series
        
        CS 5800: Dynamic programming - uses previous EMA value
        
        Args:
            prices: Series of prices
            span: Span for EMA calculation
            adjust: Whether to use adjusted calculation
            
        Returns:
            Series of EMA values
        """
        if len(prices) < 2:
            return pd.Series(index=prices.index, dtype=float)
        
        # Using pandas built-in EMA calculation
        ema = prices.ewm(span=span, adjust=adjust, min_periods=1).mean()
        return ema
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Time Complexity: O(n) - three passes for EMAs
        Space Complexity: O(n) for three series
        
        CS 5800: Combines multiple algorithms for complex analysis
        
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < 26:
            return {
                'macd': pd.Series(index=prices.index, dtype=float),
                'signal': pd.Series(index=prices.index, dtype=float),
                'histogram': pd.Series(index=prices.index, dtype=float)
            }
        
        # Calculate EMAs
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        # MACD line
        macd_line = ema_12 - ema_26
        
        # Signal line (9-day EMA of MACD)
        signal_line = self.calculate_ema(macd_line, 9)
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def generate_ma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive moving average signals.
        
        Time Complexity: O(n*m) where m is number of indicators
        Space Complexity: O(n*m) for all indicators
        
        CS 5800: Parallel algorithm design - each indicator independent
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure we have Close price
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Calculate all SMAs
        for window in self.sma_windows:
            if len(df) >= window:
                df[f'SMA_{window}'] = self.calculate_sma(df['Close'], window)
            else:
                df[f'SMA_{window}'] = np.nan
        
        # Calculate all EMAs
        for span in self.ema_spans:
            if len(df) >= span:
                df[f'EMA_{span}'] = self.calculate_ema(df['Close'], span)
            else:
                df[f'EMA_{span}'] = np.nan
        
        # Calculate MACD
        macd_dict = self.calculate_macd(df['Close'])
        df['MACD'] = macd_dict['macd']
        df['MACD_Signal'] = macd_dict['signal']
        df['MACD_Histogram'] = macd_dict['histogram']
        
        # Initialize signal columns with default values
        df['Golden_Cross_Signal'] = 0.0
        df['Price_SMA20_Signal'] = 0.0
        df['EMA_Cross_Signal'] = 0.0
        df['MACD_Buy_Signal'] = 0.0
        df['MACD_Sell_Signal'] = 0.0
        
        # Generate Golden/Death Cross signals
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            for i in range(1, len(df)):
                if pd.notna(df['SMA_50'].iloc[i]) and pd.notna(df['SMA_200'].iloc[i]):
                    # Check for Golden Cross (50 crosses above 200)
                    if pd.notna(df['SMA_50'].iloc[i-1]) and pd.notna(df['SMA_200'].iloc[i-1]):
                        if (df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i] and 
                            df['SMA_50'].iloc[i-1] <= df['SMA_200'].iloc[i-1]):
                            df.iloc[i, df.columns.get_loc('Golden_Cross_Signal')] = 1.0
                        # Death Cross (50 crosses below 200)
                        elif (df['SMA_50'].iloc[i] < df['SMA_200'].iloc[i] and 
                              df['SMA_50'].iloc[i-1] >= df['SMA_200'].iloc[i-1]):
                            df.iloc[i, df.columns.get_loc('Golden_Cross_Signal')] = -1.0
        
        # Generate Price vs SMA20 signal
        if 'SMA_20' in df.columns:
            df.loc[pd.notna(df['SMA_20']), 'Price_SMA20_Signal'] = np.where(
                df.loc[pd.notna(df['SMA_20']), 'Close'] > df.loc[pd.notna(df['SMA_20']), 'SMA_20'],
                1.0, -1.0
            )
        
        # Generate EMA crossover signal
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            df.loc[pd.notna(df['EMA_12']) & pd.notna(df['EMA_26']), 'EMA_Cross_Signal'] = np.where(
                df.loc[pd.notna(df['EMA_12']) & pd.notna(df['EMA_26']), 'EMA_12'] > 
                df.loc[pd.notna(df['EMA_12']) & pd.notna(df['EMA_26']), 'EMA_26'],
                1.0, -1.0
            )
        
        # Generate MACD signals
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            for i in range(1, len(df)):
                if pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD_Signal'].iloc[i]):
                    if pd.notna(df['MACD'].iloc[i-1]) and pd.notna(df['MACD_Signal'].iloc[i-1]):
                        # Buy signal when MACD crosses above signal line
                        if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and 
                            df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]):
                            df.iloc[i, df.columns.get_loc('MACD_Buy_Signal')] = 1.0
                        # Sell signal when MACD crosses below signal line
                        elif (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and 
                              df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]):
                            df.iloc[i, df.columns.get_loc('MACD_Sell_Signal')] = -1.0
        
        self.signals_generated = True
        return df
    
    def calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate combined signal strength from all indicators.
        
        Time Complexity: O(n) for aggregation
        Space Complexity: O(n) for output
        
        CS 5800: Weighted voting algorithm
        """
        weights = {
            'Golden_Cross_Signal': 0.25,
            'Price_SMA20_Signal': 0.20,
            'EMA_Cross_Signal': 0.25,
            'MACD_Buy_Signal': 0.15,
            'MACD_Sell_Signal': 0.15
        }
        
        signal_strength = pd.Series(index=df.index, data=0.0, dtype=float)
        
        for signal, weight in weights.items():
            if signal in df.columns:
                # Replace NaN with 0 before adding
                signal_values = df[signal].fillna(0)
                signal_strength += signal_values * weight
        
        return signal_strength
    
    def backtest_ma_strategy(self, df: pd.DataFrame, 
                            initial_capital: float = 10000) -> Dict:
        """
        Backtest moving average strategy.
        
        Time Complexity: O(n) single pass
        Space Complexity: O(n) for trade history
        """
        if not self.signals_generated:
            df = self.generate_ma_signals(df)
        
        signal_strength = self.calculate_signal_strength(df)
        
        capital = initial_capital
        shares = 0
        trades = []
        
        # Start after enough data for indicators
        # Fixed: Properly handle the index comparison
        if 'SMA_200' in df.columns:
            first_valid = df['SMA_200'].first_valid_index()
            if first_valid is not None:
                # Convert timestamp to integer position
                first_valid_pos = df.index.get_loc(first_valid)
                start_index = max(200, first_valid_pos)
            else:
                start_index = 200
        else:
            start_index = 50
        
        # Ensure start_index doesn't exceed dataframe length
        start_index = min(start_index, len(df) - 1)
        
        for i in range(start_index, len(df)):
            signal = signal_strength.iloc[i]
            price = df['Close'].iloc[i]
            
            if signal > 0.3 and capital >= price:  # Buy
                shares_to_buy = int(capital // price)
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    capital -= shares_to_buy * price
                    trades.append({
                        'date': df.index[i],
                        'type': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'signal_strength': signal
                    })
            
            elif signal < -0.3 and shares > 0:  # Sell
                capital += shares * price
                trades.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': price,
                    'shares': shares,
                    'signal_strength': signal
                })
                shares = 0
        
        # Final portfolio value
        final_value = capital + shares * df['Close'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'trades': trades
        }

# Main execution
if __name__ == "__main__":
    import os
    
    # Create directories if they don't exist
    os.makedirs('data/signals', exist_ok=True)
    
    # Load preprocessed data
    try:
        df = pd.read_csv('data/processed/AAPL_train.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        # If preprocessed doesn't exist, try raw data
        try:
            df = pd.read_csv('data/raw/AAPL_historical.csv', index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print("Error: No data file found. Please run data_collector.py first.")
            exit(1)
    
    # Initialize analyzer
    analyzer = MovingAverageAnalyzer()
    
    # Generate all signals
    print("Generating moving average signals...")
    df_with_signals = analyzer.generate_ma_signals(df)
    
    # Check if signals were generated
    print("\nSignal columns created:")
    signal_cols = ['Golden_Cross_Signal', 'Price_SMA20_Signal', 'EMA_Cross_Signal', 
                   'MACD_Buy_Signal', 'MACD_Sell_Signal']
    for col in signal_cols:
        if col in df_with_signals.columns:
            non_zero = (df_with_signals[col] != 0).sum()
            print(f"  {col}: {non_zero} non-zero signals")
    
    # Calculate signal strength
    signal_strength = analyzer.calculate_signal_strength(df_with_signals)
    print(f"\nSignal strength range: [{signal_strength.min():.2f}, {signal_strength.max():.2f}]")
    
    # Backtest strategy
    print("\nBacktesting moving average strategy...")
    results = analyzer.backtest_ma_strategy(df_with_signals)
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Save signals
    df_with_signals.to_csv('data/signals/AAPL_ma_signals.csv')
    print("\nSignals saved to data/signals/AAPL_ma_signals.csv")