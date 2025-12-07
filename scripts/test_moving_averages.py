import pandas as pd
import numpy as np
from moving_averages import MovingAverageAnalyzer

# Create sample data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
prices = np.random.uniform(100, 200, 100)
df = pd.DataFrame({'Close': prices}, index=dates)

# Initialize analyzer
analyzer = MovingAverageAnalyzer()

# Test the generate_ma_signals method
try:
    result_df = analyzer.generate_ma_signals(df)
    print("generate_ma_signals executed successfully.")
    print(f"Original df shape: {df.shape}")
    print(f"Result df shape: {result_df.shape}")
    print("Signal columns added:")
    signal_cols = [col for col in result_df.columns if 'Signal' in col]
    print(signal_cols)
    print("Sample signal values:")
    for col in signal_cols:
        print(f"{col}: {result_df[col].iloc[-5:].values}")
except Exception as e:
    print(f"Error executing generate_ma_signals: {e}")
