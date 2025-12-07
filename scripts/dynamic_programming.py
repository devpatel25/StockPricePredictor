"""
Dynamic Programming Stock Trading Optimization
CS 5800 - Algorithms Final Project

Solves the k-transactions stock problem using DP.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class DynamicProgrammingOptimizer:
    """
    Implements dynamic programming solutions for stock trading.
    CS 5800 Reference: Dynamic Programming (Module 5)
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self.dp_table = None
        self.transactions = []
        
    def max_profit_k_transactions(self, prices: List[float], 
                                 k: int) -> Tuple[float, List[Dict]]:
        """
        Find maximum profit with at most k transactions.
        
        Time Complexity: O(n*k) where n is number of days
        Space Complexity: O(k) optimized from O(n*k)
        
        CS 5800: Classic DP problem with state optimization
        
        State definition:
        - dp[i][j][0] = max profit on day i with at most j transactions, not holding stock
        - dp[i][j][1] = max profit on day i with at most j transactions, holding stock
        
        Recurrence relation:
        - dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
        - dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
        """
        n = len(prices)
        if n <= 1 or k == 0:
            return 0, []
        
        # If k >= n/2, it's unlimited transactions
        if k >= n // 2:
            return self.max_profit_unlimited(prices)
        
        # Initialize DP arrays (space optimized)
        buy = [-float('inf')] * (k + 1)
        sell = [0] * (k + 1)
        
        # Track transactions for reconstruction
        buy_days = [[] for _ in range(k + 1)]
        sell_days = [[] for _ in range(k + 1)]
        
        for i, price in enumerate(prices):
            # Process in reverse order to avoid overwriting
            for j in range(k, 0, -1):
                # Sell decision
                if sell[j] < buy[j] + price:
                    sell[j] = buy[j] + price
                    if j <= len(sell_days):
                        sell_days[j] = sell_days[j][:] 
                        if buy_days[j]:
                            sell_days[j].append((buy_days[j][-1], i))
                
                # Buy decision
                if buy[j] < sell[j-1] - price:
                    buy[j] = sell[j-1] - price
                    if j <= len(buy_days):
                        buy_days[j] = buy_days[j-1][:]
                        buy_days[j].append(i)
        
        # Reconstruct transactions
        transactions = self.reconstruct_transactions(sell_days[k])
        
        return sell[k], transactions
    
    def max_profit_unlimited(self, prices: List[float]) -> Tuple[float, List[Dict]]:
        """
        Maximum profit with unlimited transactions.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        CS 5800: Greedy algorithm - buy before every rise
        """
        profit = 0
        transactions = []
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                daily_profit = prices[i] - prices[i-1]
                profit += daily_profit
                transactions.append({
                    'buy_day': i-1,
                    'buy_price': prices[i-1],
                    'sell_day': i,
                    'sell_price': prices[i],
                    'profit': daily_profit
                })
        
        return profit, transactions
    
    def reconstruct_transactions(self, transaction_pairs: List[Tuple]) -> List[Dict]:
        """
        Reconstruct transaction details from day pairs.
        
        Time Complexity: O(k)
        Space Complexity: O(k)
        """
        transactions = []
        for buy_day, sell_day in transaction_pairs:
            if isinstance(buy_day, int) and isinstance(sell_day, int):
                transactions.append({
                    'buy_day': buy_day,
                    'sell_day': sell_day
                })
        return transactions
    
    def max_profit_with_cooldown(self, prices: List[float]) -> float:
        """
        Maximum profit with cooldown period.
        
        After selling, must wait one day before buying again.
        
        Time Complexity: O(n)
        Space Complexity: O(1) optimized from O(n)
        
        CS 5800: State machine DP
        
        States:
        - held: holding stock
        - sold: just sold, in cooldown
        - rest: can buy
        """
        if len(prices) <= 1:
            return 0
        
        # Initialize states
        held = -prices[0]  # Bought on day 0
        sold = 0           # Cannot sell on day 0
        rest = 0           # No stock, no cooldown
        
        for price in prices[1:]:
            prev_held = held
            prev_sold = sold
            prev_rest = rest
            
            held = max(prev_held, prev_rest - price)  # Hold or buy
            sold = prev_held + price                   # Sell
            rest = max(prev_rest, prev_sold)          # Rest or end cooldown
        
        return max(sold, rest)
    
    def max_profit_with_fee(self, prices: List[float], 
                           fee: float) -> Tuple[float, int]:
        """
        Maximum profit with transaction fee.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        CS 5800: Modified DP with cost consideration
        """
        if len(prices) <= 1:
            return 0, 0
        
        # States: cash (not holding) and hold (holding stock)
        cash = 0
        hold = -prices[0]
        transactions = 0
        
        for price in prices[1:]:
            prev_cash = cash
            prev_hold = hold
            
            # Either keep cash or sell stock (minus fee)
            cash = max(prev_cash, prev_hold + price - fee)
            
            # Either keep holding or buy stock
            hold = max(prev_hold, prev_cash - price)
            
            # Count transaction if we sold
            if cash != prev_cash:
                transactions += 1
        
        return cash, transactions
    
    def find_optimal_entry_exit(self, prices: pd.Series, 
                               window: int = 30) -> List[Dict]:
        """
        Find optimal entry and exit points in rolling windows.
        
        CS 5800: Sliding window with DP optimization
        """
        signals = []
        
        for i in range(window, len(prices), window // 2):
            window_prices = prices[i-window:i].values.tolist()
            
            # Find optimal 2 transactions in this window
            profit, transactions = self.max_profit_k_transactions(window_prices, 2)
            
            if transactions:
                for trans in transactions:
                    if 'buy_day' in trans and 'sell_day' in trans:
                        signals.append({
                            'window_start': i - window,
                            'buy_index': i - window + trans['buy_day'],
                            'sell_index': i - window + trans['sell_day'],
                            'expected_profit': profit
                        })
        
        return signals
    
    def dp_signal_generator(self, current_index: int, 
                           optimal_signals: List[Dict]) -> float:
        """
        Generate signal based on DP optimization.
        
        Returns:
            1.0 for buy, -1.0 for sell, 0.0 for hold
        """
        for signal in optimal_signals:
            if current_index == signal.get('buy_index'):
                return 1.0
            elif current_index == signal.get('sell_index'):
                return -1.0
        
        return 0.0
    
    def compare_strategies(self, prices: List[float]) -> pd.DataFrame:
        """
        Compare different DP strategies.
        
        CS 5800: Algorithm comparison for optimization
        """
        results = []
        
        # Strategy 1: One transaction
        profit_1, trans_1 = self.max_profit_k_transactions(prices, 1)
        results.append({
            'Strategy': '1 Transaction',
            'Max Profit': profit_1,
            'Transactions': len(trans_1),
            'Complexity': 'O(n)'
        })
        
        # Strategy 2: Two transactions
        profit_2, trans_2 = self.max_profit_k_transactions(prices, 2)
        results.append({
            'Strategy': '2 Transactions',
            'Max Profit': profit_2,
            'Transactions': len(trans_2),
            'Complexity': 'O(2n)'
        })
        
        # Strategy 3: Unlimited transactions
        profit_inf, trans_inf = self.max_profit_unlimited(prices)
        results.append({
            'Strategy': 'Unlimited',
            'Max Profit': profit_inf,
            'Transactions': len(trans_inf),
            'Complexity': 'O(n)'
        })
        
        # Strategy 4: With cooldown
        profit_cool = self.max_profit_with_cooldown(prices)
        results.append({
            'Strategy': 'With Cooldown',
            'Max Profit': profit_cool,
            'Transactions': 'N/A',
            'Complexity': 'O(n)'
        })
        
        # Strategy 5: With transaction fee ($5)
        profit_fee, num_trans = self.max_profit_with_fee(prices, 5)
        results.append({
            'Strategy': 'With $5 Fee',
            'Max Profit': profit_fee,
            'Transactions': num_trans,
            'Complexity': 'O(n)'
        })
        
        return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    # Load price data
    df = pd.read_csv('data/processed/AAPL_train.csv', 
                     index_col='Date', parse_dates=True)
    
    # Initialize optimizer
    optimizer = DynamicProgrammingOptimizer()
    
    # Test with last 100 days
    prices = df['Close'].iloc[-100:].values.tolist()
    
    print("Dynamic Programming Stock Trading Optimization")
    print("=" * 50)
    
    # Find maximum profit with different k values
    for k in [1, 2, 3, 5]:
        profit, transactions = optimizer.max_profit_k_transactions(prices, k)
        print(f"\nMax profit with {k} transactions: ${profit:.2f}")
        print(f"Number of transactions executed: {len(transactions)}")
    
    # Compare all strategies
    print("\n" + "=" * 50)
    print("Strategy Comparison:")
    comparison_df = optimizer.compare_strategies(prices)
    print(comparison_df.to_string())
    
    # Find optimal entry/exit points
    print("\n" + "=" * 50)
    print("Finding optimal entry/exit points...")
    signals = optimizer.find_optimal_entry_exit(df['Close'], window=30)
    
    print(f"Found {len(signals)} optimal trading opportunities")
    
    # Generate DP-based signals for entire dataset
    dp_signals = []
    for i in range(len(df)):
        signal = optimizer.dp_signal_generator(i, signals)
        dp_signals.append(signal)
    
    df['DP_Signal'] = dp_signals
    
    # Count signals
    buy_signals = sum(1 for s in dp_signals if s > 0)
    sell_signals = sum(1 for s in dp_signals if s < 0)
    
    print(f"\nSignal Summary:")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    print(f"Hold signals: {len(dp_signals) - buy_signals - sell_signals}")
    
    # Save signals
    df.to_csv('data/signals/AAPL_dp_signals.csv')
    print("\nDP signals saved to data/signals/AAPL_dp_signals.csv")