"""
Moving Average Algorithms Implementation
CS 5800 - Algorithms Final Project

Implements SMA, EMA, and MACD with signal generation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

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
        # Using pandas rolling for O(n) complexity instead of naive O(n*w)
        sma = prices.rolling(window=window).mean()
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
        # Alpha (smoothing factor)
        alpha = 2 / (span + 1)
        
        # Method 1: Using pandas built-in (optimized)
        ema = prices.ewm(span=span, adjust=adjust).mean()
        
        # Method 2: Manual calculation to show algorithm
        # ema_manual = [prices.iloc[0]]
        # for i in range(1, len(prices)):
        #     value = alpha * prices.iloc[i] + (1 - alpha) * ema_manual[-1]
        #     ema_manual.append(value)
        
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
    
    def detect_golden_cross(self, prices: pd.Series) -> pd.Series:
        """
        Detect Golden Cross (50-day SMA crosses above 200-day SMA).
        
        Time Complexity: O(n) for SMA calculations + O(n) for comparison
        Space Complexity: O(n) for signals
        
        CS 5800: Pattern matching in time series
        """
        sma_50 = self.calculate_sma(prices, 50)
        sma_200 = self.calculate_sma(prices, 200)
        
        # Create signal series
        signals = pd.Series(index=prices.index, dtype=float)
        signals[:] = 0.0
        
        # Detect crossovers
        for i in range(1, len(prices)):
            if pd.notna(sma_50.iloc[i]) and pd.notna(sma_200.iloc[i]):
                # Golden Cross: 50 crosses above 200
                if (sma_50.iloc[i] > sma_200.iloc[i] and 
                    sma_50.iloc[i-1] <= sma_200.iloc[i-1]):
                    signals.iloc[i] = 1.0  # Buy signal
                    
                # Death Cross: 50 crosses below 200
                elif (sma_50.iloc[i] < sma_200.iloc[i] and 
                      sma_50.iloc[i-1] >= sma_200.iloc[i-1]):
                    signals.iloc[i] = -1.0  # Sell signal
        
        return signals
    
    def generate_ma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive moving average signals.
        
        Time Complexity: O(n*m) where m is number of indicators
        Space Complexity: O(n*m) for all indicators
        
        CS 5800: Parallel algorithm design - each indicator independent
        """
        # Calculate all SMAs
        for window in self.sma_windows:
            df[f'SMA_{window}'] = self.calculate_sma(df['Close'], window)
        
        # Calculate all EMAs
        for span in self.ema_spans:
            df[f'EMA_{span}'] = self.calculate_ema(df['Close'], span)
        
        # Calculate MACD
        macd_dict = self.calculate_macd(df['Close'])
        df['MACD'] = macd_dict['macd']
        df['MACD_Signal'] = macd_dict['signal']
        df['MACD_Histogram'] = macd_dict['histogram']
        
        # Generate trading signals
        df['Golden_Cross_Signal'] = self.detect_golden_cross(df['Close'])
        
        # Price vs SMA20 signal
        df['Price_SMA20_Signal'] = np.where(
            df['Close'] > df['SMA_20'], 1, -1
        )
        
        # EMA crossover signal
        df['EMA_Cross_Signal'] = np.where(
            df['EMA_12'] > df['EMA_26'], 1, -1
        )
        
        # MACD signal
        df['MACD_Buy_Signal'] = np.where(
            (df['MACD'] > df['MACD_Signal']) & 
            (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 
            1, 0
        )
        
        df['MACD_Sell_Signal'] = np.where(
            (df['MACD'] < df['MACD_Signal']) & 
            (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), 
            -1, 0
        )
        
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
        
        signal_strength = pd.Series(index=df.index, dtype=float)
        signal_strength[:] = 0.0
        
        for signal, weight in weights.items():
            if signal in df.columns:
                signal_strength += df[signal] * weight
        
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
        
        for i in range(200, len(df)):  # Start after SMA_200 is available
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
    # Load preprocessed data
    df = pd.read_csv('data/processed/AAPL_train.csv', index_col='Date', parse_dates=True)
    
    # Initialize analyzer
    analyzer = MovingAverageAnalyzer()
    
    # Generate all signals
    print("Generating moving average signals...")
    df = analyzer.generate_ma_signals(df)
    
    # Calculate signal strength
    signal_strength = analyzer.calculate_signal_strength(df)
    
    # Backtest strategy
    print("\nBacktesting moving average strategy...")
    results = analyzer.backtest_ma_strategy(df)
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Save signals
    df.to_csv('data/signals/AAPL_ma_signals.csv')
    print("\nSignals saved to data/signals/AAPL_ma_signals.csv")