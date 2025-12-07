"""
Combined Signal Generation System
CS 5800 - Algorithms Final Project

Combines all algorithms to generate final trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class SignalGenerator:
    """
    Generates weighted trading signals from multiple algorithms.
    CS 5800 Reference: Ensemble methods and voting algorithms
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize signal generator with algorithm weights.
        
        Args:
            weights: Dictionary of algorithm weights (must sum to 1.0)
        """
        if weights is None:
            # Default weights
            self.weights = {
                'moving_average': 0.25,
                'regression': 0.30,
                'dynamic_programming': 0.20,
                'macd': 0.25
            }
        else:
            # Validate weights sum to 1.0
            if abs(sum(weights.values()) - 1.0) > 0.001:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
        
        self.signal_history = []
        self.confidence_threshold = {
            'STRONG_BUY': 0.5,
            'BUY': 0.2,
            'HOLD_HIGH': 0.1,
            'HOLD_LOW': -0.1,
            'SELL': -0.2,
            'STRONG_SELL': -0.5
        }
    
    def normalize_signal(self, signal_value: float, 
                        signal_type: str) -> float:
        """
        Normalize different signal types to [-1, 1] range.
        
        CS 5800: Normalization for consistent comparison
        """
        if signal_type == 'binary':
            # Already in {-1, 0, 1}
            return signal_value
        elif signal_type == 'percentage':
            # Convert percentage to [-1, 1]
            return np.clip(signal_value / 100, -1, 1)
        elif signal_type == 'continuous':
            # Assume already in reasonable range
            return np.clip(signal_value, -1, 1)
        else:
            return signal_value
    
    def calculate_ma_component(self, row: pd.Series) -> float:
        """
        Calculate moving average signal component.
        
        Time Complexity: O(1) - simple lookups
        """
        signals = []
        
        # Golden Cross signal
        if 'Golden_Cross_Signal' in row:
            signals.append(row['Golden_Cross_Signal'])
        
        # Price vs SMA20
        if 'Price_SMA20_Signal' in row:
            signals.append(row['Price_SMA20_Signal'] * 0.5)
        
        # EMA crossover
        if 'EMA_Cross_Signal' in row:
            signals.append(row['EMA_Cross_Signal'] * 0.5)
        
        if signals:
            return np.mean(signals)
        return 0.0
    
    def calculate_macd_component(self, row: pd.Series) -> float:
        """
        Calculate MACD signal component.
        """
        macd_signal = 0.0
        
        if 'MACD_Buy_Signal' in row:
            macd_signal += row['MACD_Buy_Signal']
        
        if 'MACD_Sell_Signal' in row:
            macd_signal += row['MACD_Sell_Signal']
        
        # Also consider MACD histogram
        if 'MACD_Histogram' in row and not pd.isna(row['MACD_Histogram']):
            # Normalize histogram to [-1, 1]
            hist_signal = np.tanh(row['MACD_Histogram'] / 100)
            macd_signal = 0.7 * macd_signal + 0.3 * hist_signal
        
        return macd_signal
    
    def calculate_regression_component(self, current_price: float,
                                      predicted_price: float) -> float:
        """
        Calculate regression signal component.
        
        CS 5800: Threshold-based decision
        """
        if predicted_price == 0 or current_price == 0:
            return 0.0
        
        price_change = (predicted_price - current_price) / current_price
        
        # Convert percentage change to signal strength
        if price_change > 0.02:  # >2% increase
            return 1.0
        elif price_change > 0.01:  # 1-2% increase
            return 0.5
        elif price_change < -0.02:  # >2% decrease
            return -1.0
        elif price_change < -0.01:  # 1-2% decrease
            return -0.5
        else:
            return 0.0
    
    def calculate_dp_component(self, row: pd.Series) -> float:
        """
        Calculate dynamic programming signal component.
        """
        if 'DP_Signal' in row:
            return row['DP_Signal']
        return 0.0
    
    def generate_combined_signal(self, row: pd.Series,
                                current_price: float = None,
                                predicted_price: float = None) -> Dict:
        """
        Generate final weighted signal from all components.
        """
        # Calculate individual components
        ma_signal = self.calculate_ma_component(row)
        macd_signal = self.calculate_macd_component(row)
        dp_signal = self.calculate_dp_component(row)
        
        # Regression component (if prices provided and not NaN)
        if (current_price is not None and predicted_price is not None and
            not pd.isna(current_price) and not pd.isna(predicted_price)):
            reg_signal = self.calculate_regression_component(
                current_price, predicted_price
            )
        else:
            reg_signal = 0.0
        
        # Apply weights
        weighted_signals = {
            'moving_average': ma_signal * self.weights['moving_average'],
            'macd': macd_signal * self.weights['macd'],
            'regression': reg_signal * self.weights['regression'],
            'dynamic_programming': dp_signal * self.weights['dynamic_programming']
        }
        
        # Calculate final score
        final_score = sum(weighted_signals.values())
        
        # Determine signal type
        if final_score > self.confidence_threshold['STRONG_BUY']:
            signal_type = 'STRONG_BUY'
        elif final_score > self.confidence_threshold['BUY']:
            signal_type = 'BUY'
        elif final_score > self.confidence_threshold['HOLD_HIGH']:
            signal_type = 'HOLD'
        elif final_score > self.confidence_threshold['HOLD_LOW']:
            signal_type = 'HOLD'
        elif final_score > self.confidence_threshold['SELL']:
            signal_type = 'SELL'
        else:
            signal_type = 'STRONG_SELL'
        
        # Calculate confidence (0-100%)
        confidence = min(abs(final_score) * 100, 100)
        
        result = {
            'signal': signal_type,
            'score': final_score,
            'confidence': confidence,
            'components': weighted_signals,
            'timestamp': row.name if hasattr(row, 'name') else None
        }
        
        # Store in history
        self.signal_history.append(result)
        
        return result
    
    def optimize_weights(self, historical_data: pd.DataFrame,
                        actual_returns: pd.Series) -> Dict[str, float]:
        """
        Optimize weights based on historical performance.
        
        CS 5800: Optimization problem - gradient descent approach
        """
        # Simplified optimization - adjust weights based on correlation
        # with actual returns
        
        best_weights = self.weights.copy()
        best_score = -float('inf')
        
        # Grid search over weight combinations
        weight_options = [0.15, 0.20, 0.25, 0.30, 0.35]
        
        for ma_w in weight_options:
            for reg_w in weight_options:
                for dp_w in weight_options:
                    macd_w = 1.0 - ma_w - reg_w - dp_w
                    
                    if macd_w >= 0.1 and macd_w <= 0.4:
                        test_weights = {
                            'moving_average': ma_w,
                            'regression': reg_w,
                            'dynamic_programming': dp_w,
                            'macd': macd_w
                        }
                        
                        # Calculate performance with these weights
                        # (Simplified - would need actual backtesting)
                        score = self.evaluate_weights(test_weights, 
                                                     historical_data, 
                                                     actual_returns)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = test_weights
        
        return best_weights
    
    def evaluate_weights(self, weights: Dict, 
                        data: pd.DataFrame, 
                        returns: pd.Series) -> float:
        """Helper function to evaluate weight performance."""
        # Placeholder - would implement actual backtesting
        return np.random.random()
    
    def generate_signals_for_dataframe(self, df: pd.DataFrame,
                                      predictions: pd.Series = None) -> pd.DataFrame:
        """
        Generate signals for entire dataframe.
        """
        signals = []
        scores = []
        confidences = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Resolve predicted price:
            pred_price = None
            if predictions is not None:
                # Prefer alignment by index/label (e.g. Date)
                try:
                    if isinstance(predictions, pd.Series):
                        pred_price = predictions.get(row.name, None)
                    else:
                        # fallback to positional indexing for lists/ndarrays or DataFrames
                        if hasattr(predictions, 'iloc'):
                            pred_price = predictions.iloc[i]
                        else:
                            pred_price = predictions[i]
                except Exception:
                    pred_price = None

                # normalize NaN -> None
                if pd.isna(pred_price):
                    pred_price = None

            # Current price (safe access)
            curr_price = row.get('Close', None) if isinstance(row, pd.Series) else None
            
            # Generate signal
            result = self.generate_combined_signal(row, curr_price, pred_price)
            
            signals.append(result['signal'])
            scores.append(result['score'])
            confidences.append(result['confidence'])
        
        # Add to dataframe
        df['Final_Signal'] = signals
        df['Signal_Score'] = scores
        df['Signal_Confidence'] = confidences
        
        return df
    
    def save_signal_history(self, filepath: str):
        """Save signal history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.signal_history, f, indent=2, default=str)
        print(f"Signal history saved to {filepath}")

# Main execution
if __name__ == "__main__":
    # Load all signal data
    df = pd.read_csv('data/signals/AAPL_dp_signals.csv', 
                     index_col='Date', parse_dates=True)
    
    # Load regression predictions
    predictions = pd.read_csv('data/predictions/regression_predictions.csv',
                             index_col=0)
    
    # Initialize signal generator
    generator = SignalGenerator()
    
    print("Generating combined signals...")
    print("=" * 50)
    
    # Generate signals for entire dataset
    df = generator.generate_signals_for_dataframe(df, predictions['Predicted'])
    
    # Print signal distribution
    signal_dist = df['Final_Signal'].value_counts()
    print("\nSignal Distribution:")
    print(signal_dist)
    
    # Calculate average confidence by signal type
    avg_confidence = df.groupby('Final_Signal')['Signal_Confidence'].mean()
    print("\nAverage Confidence by Signal Type:")
    print(avg_confidence)
    
    # Save final signals
    df.to_csv('data/signals/AAPL_final_signals.csv')
    print("\nFinal signals saved to data/signals/AAPL_final_signals.csv")
    
    # Save signal history
    generator.save_signal_history('data/signals/signal_history.json')
    
    print("\n" + "=" * 50)
    print("Signal generation complete!")