"""
Performance Metrics Calculator
CS 5800 - Algorithms Final Project

Calculates comprehensive performance metrics for trading strategies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

class PerformanceMetrics:
    """
    Calculates trading strategy performance metrics.
    CS 5800 Reference: Statistical algorithms (Module 6)
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple returns.
        
        Time Complexity: O(n)
        """
        return prices.pct_change()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate logarithmic returns.
        
        CS 5800: Log returns for better statistical properties
        """
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Formula: (mean_return - risk_free_rate) / std_deviation * sqrt(252)
        """
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        CS 5800: Modified risk metric focusing on downside
        """
        downside_returns = returns[returns < target_return]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return 0
        
        return (returns.mean() - target_return) / downside_std * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and dates.
        
        Time Complexity: O(n)
        CS 5800: Peak detection algorithm
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        # Find start date (peak before the trough)
        peak_date = cumulative[:end_date].idxmax()
        
        return max_dd, peak_date, end_date
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        """
        annual_return = returns.mean() * 252
        max_dd, _, _ = PerformanceMetrics.maximum_drawdown(returns)
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        CS 5800: Percentile calculation for risk management
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, 
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        """
        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_all_metrics(portfolio_values: pd.Series,
                            initial_capital: float) -> Dict:
        """
        Calculate all performance metrics.
        
        CS 5800: Comprehensive analysis pipeline
        """
        returns = portfolio_values.pct_change().dropna()
        
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        annual_return = total_return * (252 / len(portfolio_values))
        
        max_dd, peak_date, trough_date = PerformanceMetrics.maximum_drawdown(returns)
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Sharpe Ratio': f"{PerformanceMetrics.sharpe_ratio(returns):.2f}",
            'Sortino Ratio': f"{PerformanceMetrics.sortino_ratio(returns):.2f}",
            'Calmar Ratio': f"{PerformanceMetrics.calmar_ratio(returns):.2f}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Max DD Peak Date': peak_date.strftime('%Y-%m-%d') if peak_date else 'N/A',
            'Max DD Trough Date': trough_date.strftime('%Y-%m-%d') if trough_date else 'N/A',
            'VaR (95%)': f"{PerformanceMetrics.value_at_risk(returns):.2%}",
            'CVaR (95%)': f"{PerformanceMetrics.conditional_value_at_risk(returns):.2%}",
            'Daily Volatility': f"{returns.std():.2%}",
            'Annual Volatility': f"{returns.std() * np.sqrt(252):.2%}",
            'Skewness': f"{stats.skew(returns):.2f}",
            'Kurtosis': f"{stats.kurtosis(returns):.2f}"
        }
        
        return metrics

# Main execution
if __name__ == "__main__":
    # Load portfolio values
    pf_df = pd.read_csv('results/portfolio_values.csv', 
                       index_col='date', parse_dates=True)
    
    # Calculate all metrics
    metrics = PerformanceMetrics.calculate_all_metrics(
        pf_df['value'], 
        initial_capital=10000
    )
    
    print("Performance Metrics")
    print("=" * 50)
    
    for metric, value in metrics.items():
        print(f"{metric:.<25} {value}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/performance_metrics.csv', index=False)
    print("\n" + "=" * 50)
    print("Metrics saved to results/performance_metrics.csv")