import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.data_collector import StockDataCollector
from scripts.preprocessor import DataPreprocessor
from scripts.moving_averages import MovingAverageAnalyzer
from scripts.regression_predictor import RegressionPredictor
from scripts.dynamic_programming import DynamicProgrammingOptimizer
from scripts.signal_generator import SignalGenerator
from scripts.backtester import Backtester
from scripts.performance_metrics import PerformanceMetrics

def main():
    # Step 1: Collect Data
    print("Step 1: Collecting data...")
    collector = StockDataCollector(['AAPL'], '2020-01-01', '2024-11-01')
    stock_data = collector.fetch_all_stocks()
    collector.save_to_csv()
    
    # Step 2: Preprocess Data
    print("\nStep 2: Preprocessing data...")
    df = pd.read_csv('data/raw/AAPL_historical.csv', index_col='Date', parse_dates=True)
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.create_technical_features(df)
    train_df, test_df = preprocessor.prepare_train_test_split(df)
    
    # Step 3: Generate Moving Average Signals
    print("\nStep 3: Generating moving average signals...")
    ma_analyzer = MovingAverageAnalyzer()
    train_df = ma_analyzer.generate_ma_signals(train_df)
    
    # Step 4: Generate Regression Predictions
    print("\nStep 4: Training regression model...")
    predictor = RegressionPredictor(lookback_days=5)
    
    # Create features from the dataframe with MA signals
    X, y = predictor.create_features(train_df)
    
    # IMPORTANT: Remove any rows with NaN values
    print(f"Initial feature shape: {X.shape}")
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        print("Found NaN values in features, cleaning...")
        # Create a mask for rows without NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"Cleaned feature shape: {X.shape}")
    
    if len(X) > 0:
        # Split data for training and testing
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train the model
        predictor.train_linear_regression(X_train, y_train)
        
        # Generate predictions for the entire training dataframe
        predictions = []
        pred_dates = []
        
        # For each row in train_df, try to generate a prediction
        for i in range(len(train_df)):
            try:
                # Create features for this specific row
                features = []
                
                # Get the last 5 days of prices (or less if not available)
                lookback = min(5, i)
                if lookback < 5:
                    # Not enough history, use a simple prediction
                    predictions.append(train_df['Close'].iloc[i])
                else:
                    # Get price features
                    for j in range(lookback):
                        features.append(train_df['Close'].iloc[i-lookback+j])
                    
                    # Add volume feature
                    volume = train_df['Volume'].iloc[i-1] if i > 0 else train_df['Volume'].iloc[0]
                    features.append(volume / 1e9)
                    
                    # Add technical features if available
                    if 'Volatility' in train_df.columns:
                        vol = train_df['Volatility'].iloc[i-1] if i > 0 else 0
                        features.append(vol if pd.notna(vol) else 0)
                    else:
                        features.append(0)
                    
                    if 'RSI' in train_df.columns:
                        rsi = train_df['RSI'].iloc[i-1] if i > 0 else 50
                        features.append(rsi if pd.notna(rsi) else 50)
                    else:
                        features.append(50)
                    
                    if 'SMA_20' in train_df.columns:
                        sma = train_df['SMA_20'].iloc[i-1] if i > 0 else train_df['Close'].iloc[i-1]
                        features.append(1.0 if pd.notna(sma) and sma > 0 else 1.0)
                    else:
                        features.append(1.0)
                    
                    if 'EMA_12' in train_df.columns:
                        ema = train_df['EMA_12'].iloc[i-1] if i > 0 else train_df['Close'].iloc[i-1]
                        features.append(ema if pd.notna(ema) else train_df['Close'].iloc[i-1])
                    else:
                        features.append(train_df['Close'].iloc[i-1] if i > 0 else train_df['Close'].iloc[0])
                    
                    # Make prediction
                    features_array = np.array(features).reshape(1, -1)
                    
                    # Check for NaN in features
                    if np.any(np.isnan(features_array)):
                        predictions.append(train_df['Close'].iloc[i])
                    else:
                        pred = predictor.predict_next_day(predictor.models['linear'], features_array)
                        predictions.append(pred)
                
                pred_dates.append(train_df.index[i])
                
            except Exception as e:
                # If prediction fails, use current price
                predictions.append(train_df['Close'].iloc[i])
                pred_dates.append(train_df.index[i])
        
        # Add predictions to dataframe
        train_df['Predicted_Price'] = predictions
        
    else:
        print("Not enough valid data for regression training, using simple predictions")
        train_df['Predicted_Price'] = train_df['Close']
    
    # Step 5: Generate Dynamic Programming Signals
    print("\nStep 5: Generating DP signals...")
    dp_optimizer = DynamicProgrammingOptimizer()
    
    # Initialize DP_Signal column
    train_df['DP_Signal'] = 0.0
    
    # Generate DP signals using sliding window
    prices_list = train_df['Close'].values.tolist()
    window_size = 30
    
    if len(prices_list) > window_size:
        for i in range(window_size, len(prices_list), 10):
            window_prices = prices_list[i-window_size:i]
            try:
                _, transactions = dp_optimizer.max_profit_k_transactions(window_prices, 2)
                
                for trans in transactions:
                    if isinstance(trans, dict) and 'buy_day' in trans and 'sell_day' in trans:
                        buy_idx = i - window_size + trans['buy_day']
                        sell_idx = i - window_size + trans['sell_day']
                        
                        if 0 <= buy_idx < len(train_df):
                            train_df.iloc[buy_idx, train_df.columns.get_loc('DP_Signal')] = 1.0
                        if 0 <= sell_idx < len(train_df):
                            train_df.iloc[sell_idx, train_df.columns.get_loc('DP_Signal')] = -1.0
            except Exception as e:
                print(f"DP optimization failed for window at {i}: {e}")
                continue
    
    # Step 6: Generate Combined Signals
    print("\nStep 6: Generating combined signals...")
    signal_gen = SignalGenerator()
    
    # Initialize signal columns
    train_df['Final_Signal'] = 'HOLD'
    train_df['Signal_Score'] = 0.0
    train_df['Signal_Confidence'] = 0.0
    
    # Generate signals for each row
    for i in range(len(train_df)):
        try:
            row = train_df.iloc[i]
            current_price = row['Close']
            predicted_price = row.get('Predicted_Price', current_price)
            
            # Ensure predicted_price is valid
            if pd.isna(predicted_price):
                predicted_price = current_price
            
            result = signal_gen.generate_combined_signal(row, current_price, predicted_price)
            
            train_df.at[train_df.index[i], 'Final_Signal'] = result['signal']
            train_df.at[train_df.index[i], 'Signal_Score'] = result['score']
            train_df.at[train_df.index[i], 'Signal_Confidence'] = result['confidence']
        except Exception as e:
            print(f"Signal generation failed for row {i}: {e}")
            continue
    
    # Print signal distribution
    print("\nSignal Distribution:")
    print(train_df['Final_Signal'].value_counts())
    
    # Save the signals
    os.makedirs('data/signals', exist_ok=True)
    train_df.to_csv('data/signals/AAPL_final_signals.csv')
    print("\nSignals saved to data/signals/AAPL_final_signals.csv")
    
    # Step 7: Backtest
    print("\nStep 7: Running backtest...")
    backtester = Backtester(initial_capital=10000, commission=5, slippage=0.001)
    metrics = backtester.run_backtest(train_df)
    
    # Step 8: Calculate Performance Metrics
    print("\nStep 8: Calculating performance metrics...")
    if 'portfolio_values' in metrics and metrics['portfolio_values'] is not None:
        pf_df = metrics['portfolio_values']
        
        # Save portfolio values
        os.makedirs('results', exist_ok=True)
        pf_df.to_csv('results/portfolio_values.csv')
        
        # Calculate all metrics
        all_metrics = PerformanceMetrics.calculate_all_metrics(
            pf_df['value'],
            initial_capital=10000
        )
        
        print("\nPerformance Metrics:")
        print("=" * 50)
        for metric, value in all_metrics.items():
            print(f"{metric}: {value}")
        
        print("\nMetrics saved to results/performance_metrics.csv")
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        
    else:
        print("No portfolio values generated! Check if trades were executed.")
        print(f"Number of trades executed: {metrics.get('num_trades', 0)}")

if __name__ == "__main__":
    main()