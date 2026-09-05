import pytest
import pandas as pd
from src.predict import OlistPredictor

@pytest.fixture
def predictor():
    return OlistPredictor()

@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        'order_purchase_timestamp': '2018-05-10 10:00:00',
        'total_price': 150.0,
        'total_freight': 20.0,
        'total_items': 1,
        'total_payment': 170.0,
        'payment_installments': 2,
        'review_score': 5.0
    }])

def test_preprocessing_shape(predictor, sample_input):
    processed = predictor.preprocess(sample_input)
    assert processed.shape[1] == len(predictor.feature_cols)

def test_prediction_output_structure(predictor, sample_input):
    results = predictor.predict(sample_input)
    assert len(results) == 1
    assert "prediction" in results[0]
    assert "status" in results[0]
    assert "probability" in results[0]
    assert results[0]["prediction"] in [0, 1]