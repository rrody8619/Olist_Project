import pytest
import pandas as pd
from src.validation import validate_input_data

def test_validate_input_data_success():
    """Test that validation succeeds when all inputs are within allowed ranges."""
    valid_data = pd.DataFrame([{
        "order_purchase_timestamp": "2018-05-10 10:00:00",
        "total_price": 150.0,
        "total_freight": 20.0,
        "total_items": 1,
        "total_payment": 170.0,
        "payment_installments": 2,
        "review_score": 5.0
    }])
    assert validate_input_data(valid_data) is True

def test_validate_input_data_invalid_price():
    """Test that validation fails and raises ValueError when total_price is negative."""
    invalid_data = pd.DataFrame([{
        "order_purchase_timestamp": "2018-05-10 10:00:00",
        "total_price": -10.0,
        "total_freight": 20.0,
        "total_items": 1,
        "total_payment": 170.0,
        "payment_installments": 2,
        "review_score": 5.0
    }])
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_input_data(invalid_data)

def test_validate_input_data_null_timestamp():
    """Test that validation fails and raises ValueError when timestamp is None."""
    invalid_data = pd.DataFrame([{
        "order_purchase_timestamp": None,
        "total_price": 150.0,
        "total_freight": 20.0,
        "total_items": 1,
        "total_payment": 170.0,
        "payment_installments": 2,
        "review_score": 5.0
    }])
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_input_data(invalid_data)