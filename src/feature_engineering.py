import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. قراءة المجموعات الثلاث
train_df = pd.read_parquet('data/processed/train.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')
test_df = pd.read_parquet('data/processed/test.parquet')

# 2. استخراج ميزات أوقات الشراء وتأخير التسليم المتوقع
for df in [train_df, val_df, test_df]:
    df['purchase_year'] = pd.to_datetime(df['order_purchase_timestamp']).dt.year
    df['purchase_month'] = pd.to_datetime(df['order_purchase_timestamp']).dt.month
    df['purchase_dayofweek'] = pd.to_datetime(df['order_purchase_timestamp']).dt.dayofweek
    df['purchase_hour'] = pd.to_datetime(df['order_purchase_timestamp']).dt.hour

# 3. تحديد الميزات العدديّة (Numerical Features) والهدف Target
feature_cols = [
    'total_price', 'total_freight', 'total_items',
    'total_payment', 'payment_installments', 'review_score',
    'purchase_year', 'purchase_month', 'purchase_dayofweek', 'purchase_hour'
]

# 4. معالجة القيم المفقودة (Imputation) وتجهيز المقياس (Scaling)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Fit فقط على داتا الـ Train لتجنب Data Leakage
X_train_num = imputer.fit_transform(train_df[feature_cols])
X_train_scaled = scaler.fit_transform(X_train_num)

X_val_num = imputer.transform(val_df[feature_cols])
X_val_scaled = scaler.transform(X_val_num)

X_test_num = imputer.transform(test_df[feature_cols])
X_test_scaled = scaler.transform(X_test_num)

# إعادة الميزات المعالجة للـ DataFrames
train_fe = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_fe['is_late'] = train_df['is_late'].values

val_fe = pd.DataFrame(X_val_scaled, columns=feature_cols)
val_fe['is_late'] = val_df['is_late'].values

test_fe = pd.DataFrame(X_test_scaled, columns=feature_cols)
test_fe['is_late'] = test_df['is_late'].values

# 5. حفظ البيانات والـ Artifacts
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

train_fe.to_parquet('data/processed/train_fe.parquet', index=False)
val_fe.to_parquet('data/processed/val_fe.parquet', index=False)
test_fe.to_parquet('data/processed/test_fe.parquet', index=False)

joblib.dump(imputer, 'models/imputer.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_cols, 'models/feature_cols.joblib')

print("Feature engineering completed successfully.")
print("Saved transformed features to 'data/processed/' and artifacts to 'models/'")