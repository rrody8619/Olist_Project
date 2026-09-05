import pytest
import pandas as pd
from src.validation import validate_input_data

def test_validate_input_data_success():
    """اختبار نجاح البيانات عندما تكون جميع المدخلات ضمن النطاق المسموح"""
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
    """اختبار فشل الفحص ويرفع ValueError لو السعر بالسالب"""
    invalid_data = pd.DataFrame([{
        "order_purchase_timestamp": "2018-05-10 10:00:00",
        "total_price": -10.0,  # قيمة غير مسموحة
        "total_freight": 20.0,
        "total_items": 1,
        "total_payment": 170.0,
        "payment_installments": 2,
        "review_score": 5.0
    }])
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_input_data(invalid_data)

def test_validate_input_data_null_timestamp():
    """اختبار فشل الفحص لو الـ timestamp بـ None"""
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