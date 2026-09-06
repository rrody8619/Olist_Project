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
        """Load saved artifacts from the training pipeline."""
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
        """Apply preprocessing steps without refitting."""
        df = input_df.copy()

        # Extract temporal features (must match src/feature_engineering.py exactly)
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['purchase_year'] = df['order_purchase_timestamp'].dt.year
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month
        df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

        # Estimated delivery days: known at order time (the delivery estimate
        # shown to the customer at checkout), same derivation as training.
        if 'order_estimated_delivery_date' in df.columns:
            df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
            df['estimated_delivery_days'] = (
                df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
            ).dt.total_seconds() / (24 * 3600)
        else:
            raise ValueError(
                "Missing required field 'order_estimated_delivery_date' needed to compute "
                "'estimated_delivery_days' for prediction."
            )

        # Select features and apply imputer and scaler transformations
        df_features = df[self.feature_cols]
        num_imputed = self.imputer.transform(df_features)
        scaled_features = self.scaler.transform(num_imputed)

        return pd.DataFrame(scaled_features, columns=self.feature_cols)

    def predict(self, input_df: pd.DataFrame) -> dict:
        """Predict order status along with its corresponding probability."""
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
        'order_estimated_delivery_date': '2018-05-20 00:00:00',
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