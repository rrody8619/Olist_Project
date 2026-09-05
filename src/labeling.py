import os
import pandas as pd


input_path = 'data/processed/ml_table.parquet'
df = pd.read_parquet(input_path)


date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])


df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

output_path = 'data/processed/labeled_table.parquet'
df.to_parquet(output_path, index=False)

print(f"Labeling completed successfully. Output saved to {output_path}")