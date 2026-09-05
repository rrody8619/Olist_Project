import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. قراءة البيانات الجاهزة من data/processed
train_df = pd.read_parquet('data/processed/train_fe.parquet')
val_df = pd.read_parquet('data/processed/val_fe.parquet')

# 2. قراءة قائمة الأعمدة المستخدمة من المجلد القياسي models
feature_cols = joblib.load('models/feature_cols.joblib')

X_train = train_df[feature_cols]
y_train = train_df['is_late']

X_val = val_df[feature_cols]
y_val = val_df['is_late']

# 3. إنشاء الموديل وتدريبه
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print("--- Evaluation Report (Validation Set) ---")
print(classification_report(y_val, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_val, y_proba):.4f}")


os.makedirs('models', exist_ok=True)
model_path = 'models/model.joblib'
joblib.dump(model, model_path)

print(f"\nModel trained and saved successfully to '{model_path}'")