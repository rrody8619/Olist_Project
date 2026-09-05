import joblib
import pandas as pd
import yaml
from src.logger import logger

class OlistPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model_version = self.config["model"]["version"]
        self._load_artifacts()

    def _load_artifacts(self):
        """تحميل الكائنات المحفوظة من مرحلة التدريب"""
        try:
            self.imputer = joblib.load(self.config["paths"]["imputer_path"])
            self.scaler = joblib.load(self.config["paths"]["scaler_path"])
            self.feature_cols = joblib.load(self.config["paths"]["features_path"])
            self.model = joblib.load(self.config["paths"]["model_path"])
            logger.info(f"Artifacts and Model v{self.model_version} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            raise e

    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """تطبيق Preprocessing بدون إعادة Fit"""
        df = input_df.copy()

        # استخراج الميزات الزمنية
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['purchase_year'] = df['order_purchase_timestamp'].dt.year
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month
        df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

        # اختيار الميزات وتمريرها على Imputer & Scaler
        df_features = df[self.feature_cols]
        num_imputed = self.imputer.transform(df_features)
        scaled_features = self.scaler.transform(num_imputed)

        return pd.DataFrame(scaled_features, columns=self.feature_cols)

    def predict(self, input_df: pd.DataFrame) -> dict:
        """التنبؤ بحالة الطلب مع الاحتمالية"""
        processed_data = self.preprocess(input_df)
        
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]

        results = []
        for pred, proba in zip(predictions, probabilities):
            status = "Late" if pred == 1 else "On Time"
            results.append({
                "prediction": int(pred),
                "status": status,
                "probability": float(proba),
                "model_version": self.model_version
            })
            
        logger.info(f"Processed {len(results)} prediction requests successfully.")
        return results

if __name__ == "__main__":
    sample_order = pd.DataFrame([{
        'order_purchase_timestamp': '2018-05-10 10:00:00',
        'total_price': 150.0,
        'total_freight': 20.0,
        'total_items': 1,
        'total_payment': 170.0,
        'payment_installments': 2,
        'review_score': 5.0
    }])
    
    predictor = OlistPredictor()
    output = predictor.predict(sample_order)
    print("Sample Prediction Output:", output)