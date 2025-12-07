"""
Backtesting Framework
CS 5800 - Algorithms Final Project

Comprehensive backtesting with realistic constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class Backtester:
    """
    Backtests trading strategies with realistic constraints.
    CS 5800 Reference: Simulation algorithms (Module 8)
    """
    
    def __init__(self, initial_capital: float = 10000,
                commission: float = 5.0,
                slippage: float = 0.001):
        """
        Initialize backtester with parameters.
        
        Args:
            initial_capital: Starting capital
            commission: Fixed commission per trade
            slippage: Slippage percentage (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.shares = 0
        self.trades = []
        self.portfolio_values = []
        self.positions = []
    
    def calculate_trade_cost(self, price: float, shares: int) -> float:
        """
        Calculate total cost including commission and slippage.
        
        CS 5800: Cost function optimization
        """
        base_cost = price * shares
        slippage_cost = base_cost * self.slippage
        total_cost = base_cost + slippage_cost + self.commission
        return total_cost
    
    def execute_buy(self, date: pd.Timestamp, price: float, 
                   signal_strength: float) -> Dict:
        """
        Execute buy order with position sizing.
        
        CS 5800: Resource allocation algorithm
        """
        # Position sizing based on signal strength
        position_size = min(signal_strength, 1.0)
        available_capital = self.capital - self.commission
        
        if available_capital <= 0:
            return None
        
        # Calculate shares to buy
        max_shares = int(available_capital / (price * (1 + self.slippage)))
        shares_to_buy = int(max_shares * position_size)
        
        if shares_to_buy == 0:
            return None
        
        # Execute trade
        total_cost = self.calculate_trade_cost(price, shares_to_buy)
        
        if total_cost <= self.capital:
            self.capital -= total_cost
            self.shares += shares_to_buy
            
            trade = {
                'date': date,
                'type': 'BUY',
                'price': price,
                'shares': shares_to_buy,
                'cost': total_cost,
                'capital_after': self.capital,
                'total_shares': self.shares,
                'signal_strength': signal_strength
            }
            
            self.trades.append(trade)
            return trade
        
        return None
    
    def execute_sell(self, date: pd.Timestamp, price: float,
                    signal_strength: float) -> Dict:
        """
        Execute sell order.
        
        CS 5800: Exit strategy optimization
        """
        if self.shares == 0:
            return None
        
        # Partial or full sell based on signal strength
        sell_percentage = min(abs(signal_strength), 1.0)
        shares_to_sell = int(self.shares * sell_percentage)
        
        if shares_to_sell == 0:
            return None
        
        # Calculate proceeds
        base_proceeds = price * shares_to_sell
        slippage_cost = base_proceeds * self.slippage
        net_proceeds = base_proceeds - slippage_cost - self.commission
        
        # Execute trade
        self.capital += net_proceeds
        self.shares -= shares_to_sell
        
        trade = {
            'date': date,
            'type': 'SELL',
            'price': price,
            'shares': shares_to_sell,
            'proceeds': net_proceeds,
            'capital_after': self.capital,
            'total_shares': self.shares,
            'signal_strength': signal_strength
        }
        
        self.trades.append(trade)
        return trade
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run complete backtest on dataframe with signals.
        
        Time Complexity: O(n) where n is number of days
        Space Complexity: O(n) for storing trades
        
        CS 5800: Event-driven simulation
        """
        self.reset()
        
        for i in range(len(df)):
            row = df.iloc[i]
            date = row.name
            price = row['Close']
            
            # Get signal
            signal = row.get('Final_Signal', 'HOLD')
            signal_score = row.get('Signal_Score', 0)
            
            # Execute trades based on signal
            if signal in ['BUY', 'STRONG_BUY']:
                self.execute_buy(date, price, abs(signal_score))
            elif signal in ['SELL', 'STRONG_SELL']:
                self.execute_sell(date, price, abs(signal_score))
            
            # Calculate portfolio value
            portfolio_value = self.capital + self.shares * price
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'capital': self.capital,
                'shares': self.shares,
                'stock_value': self.shares * price
            })
        
        return self.calculate_metrics(df)
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        CS 5800: Statistical analysis algorithms
        """
        if not self.portfolio_values:
            return {}
        
        # Convert to DataFrame for easier calculation
        pf_df = pd.DataFrame(self.portfolio_values)
        pf_df.set_index('date', inplace=True)
        
        # Basic metrics
        final_value = pf_df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        pf_df['returns'] = pf_df['value'].pct_change()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = pf_df['returns'].mean() / pf_df['returns'].std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = (1 + pf_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = 0
        losing_trades = 0
        
        for i in range(0, len(self.trades), 2):  # Pair buy and sell
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                if sell_trade['type'] == 'SELL':
                    buy_price = buy_trade['price']
                    sell_price = sell_trade['price']
                    
                    if sell_price > buy_price:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Calculate buy-and-hold comparison
        buy_hold_shares = self.initial_capital / df['Close'].iloc[0]
        buy_hold_value = buy_hold_shares * df['Close'].iloc[-1]
        buy_hold_return = (buy_hold_value - self.initial_capital) / self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': total_return * (252 / len(df)),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'portfolio_values': pf_df
        }
    
    def plot_results(self, metrics: Dict):
        """
        Visualize backtest results.
        
        CS 5800: Data visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        pf_df = metrics['portfolio_values']
        axes[0, 0].plot(pf_df.index, pf_df['value'], label='Portfolio Value')
        axes[0, 0].axhline(y=self.initial_capital, color='r', 
                          linestyle='--', label='Initial Capital')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Daily returns distribution
        axes[0, 1].hist(pf_df['returns'].dropna(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Drawdown
        cumulative = (1 + pf_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, 
                               alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)
        
        # Trade distribution
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        axes[1, 1].bar(['Buy', 'Sell'], [len(buy_trades), len(sell_trades)])
        axes[1, 1].set_title('Trade Distribution')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/backtest_results.png')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data with signals
    df = pd.read_csv('data/signals/AAPL_final_signals.csv',
                     index_col='Date', parse_dates=True)
    
    # Initialize backtester
    backtester = Backtester(initial_capital=10000, 
                           commission=5, 
                           slippage=0.001)
    
    print("Running backtest...")
    print("=" * 50)
    
    # Run backtest
    metrics = backtester.run_backtest(df)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
    print(f"Excess Return: {metrics['excess_return']:.2%}")
    
    # Plot results
    print("\nGenerating plots...")
    backtester.plot_results(metrics)
    
    # Save detailed trade log
    trades_df = pd.DataFrame(backtester.trades)
    trades_df.to_csv('results/trade_log.csv', index=False)
    print("\nTrade log saved to results/trade_log.csv")
    
    # Save portfolio values
    metrics['portfolio_values'].to_csv('results/portfolio_values.csv')
    print("Portfolio values saved to results/portfolio_values.csv")