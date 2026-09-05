import os
import pandas as pd


input_path = 'data/processed/labeled_table.parquet'
df = pd.read_parquet(input_path)

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])


df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)

n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]


os.makedirs('data/processed', exist_ok=True)

train_df.to_parquet('data/processed/train.parquet', index=False)
val_df.to_parquet('data/processed/val.parquet', index=False)
test_df.to_parquet('data/processed/test.parquet', index=False)

print("Data splitting completed successfully:")
print(f" - Train shape: {train_df.shape}")
print(f" - Val shape:   {val_df.shape}")
print(f" - Test shape:  {test_df.shape}")