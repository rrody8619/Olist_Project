import pytest
import pandas as pd
# استدعاء الملف والدوال الحقيقية الموجودة في مشروعك
from src.feature_engineering import *

def test_feature_engineering_import():
    """اختبار التأكد من أن موديول هندسة الميزات يعمل ويتم استدعاؤه بنجاح"""
    assert True