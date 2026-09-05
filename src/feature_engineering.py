import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def extract_base_features(df):
    data = df.copy()

    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    if 'order_purchase_timestamp' in data.columns:
        data['purchase_year'] = data['order_purchase_timestamp'].dt.year
        data['purchase_month'] = data['order_purchase_timestamp'].dt.month
        data['purchase_dayofweek'] = data['order_purchase_timestamp'].dt.dayofweek
        data['purchase_hour'] = data['order_purchase_timestamp'].dt.hour

    if 'order_estimated_delivery_date' in data.columns and 'order_purchase_timestamp' in data.columns:
        data['estimated_delivery_days'] = (
            data['order_estimated_delivery_date'] - data['order_purchase_timestamp']
        ).dt.total_seconds() / (24 * 3600)

    return data


def main():
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    train_fe = extract_base_features(train_df)
    val_fe = extract_base_features(val_df)
    test_fe = extract_base_features(test_df)

    feature_cols = [
        'purchase_year', 'purchase_month', 'purchase_dayofweek', 'purchase_hour',
        'estimated_delivery_days', 'total_price', 'total_freight',
        'total_items', 'total_payment', 'payment_installments'
    ]

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    train_fe[feature_cols] = imputer.fit_transform(train_fe[feature_cols])
    train_fe[feature_cols] = scaler.fit_transform(train_fe[feature_cols])

    # Transform فقط على Validation و Test باستعمال نفس الكائنات المجهزة
    val_fe[feature_cols] = imputer.transform(val_fe[feature_cols])
    val_fe[feature_cols] = scaler.transform(val_fe[feature_cols])

    test_fe[feature_cols] = imputer.transform(test_fe[feature_cols])
    test_fe[feature_cols] = scaler.transform(test_fe[feature_cols])

    os.makedirs('data/processed', exist_ok=True)
    train_fe.to_parquet('data/processed/train_fe.parquet', index=False)
    val_fe.to_parquet('data/processed/val_fe.parquet', index=False)
    test_fe.to_parquet('data/processed/test_fe.parquet', index=False)

    os.makedirs('models', exist_ok=True)
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_cols, 'models/feature_cols.joblib')

    print("feature_engineering: done")
    print(f"train_fe: {train_fe.shape}, val_fe: {val_fe.shape}, test_fe: {test_fe.shape}")


if __name__ == '__main__':
    main()