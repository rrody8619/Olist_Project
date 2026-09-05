import sys
from pathlib import Path
from typing import List
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# إضافة مجلد الجذر إلى sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import OlistPredictor
from src.logger import logger
from src.validation import validate_input_data

# إعداد تطبيق FastAPI
app = FastAPI(
    title="Olist Order Delivery Prediction API",
    description="Inference service to predict whether an Olist order will arrive late or on time.",
    version="1.0.0"
)

# تحميل المحرك مرة واحدة عند تشغيل الخدمة
predictor = OlistPredictor()

# Schema للتحقق من مدخلات الطلب (متوافقة مع Pydantic V2)
class OrderSchema(BaseModel):
    order_purchase_timestamp: str = Field(..., json_schema_extra={"example": "2018-05-10 10:00:00"})
    total_price: float = Field(..., gt=0, json_schema_extra={"example": 150.0})
    total_freight: float = Field(..., ge=0, json_schema_extra={"example": 20.0})
    total_items: int = Field(..., gt=0, json_schema_extra={"example": 1})
    total_payment: float = Field(..., gt=0, json_schema_extra={"example": 170.0})
    payment_installments: int = Field(..., ge=1, json_schema_extra={"example": 2})
    review_score: float = Field(..., ge=1.0, le=5.0, json_schema_extra={"example": 5.0})

# Schema للاستجابة
class PredictionOutput(BaseModel):
    prediction: int
    status: str
    probability: float
    model_version: str

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "Olist Inference API"}

@app.get("/model-info", status_code=status.HTTP_200_OK)
def model_info():
    return {
        "model_name": predictor.config["model"]["name"],
        "model_version": predictor.model_version,
        "features": predictor.feature_cols
    }

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict_single(order: OrderSchema):
    try:
        start_time = datetime.utcnow()
        # تحويل البيانات باستخدام model_dump() بدلاً من dict()
        df = pd.DataFrame([order.model_dump()])
        
        # 1. طبقة التحقق من صحة البيانات بـ Great Expectations
        validate_input_data(df)
        
        # 2. التنبؤ
        results = predictor.predict(df)
        prediction_res = results[0]

        # 3. تسجيل المراقبة (Prediction Logging)
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Prediction Success | Input: {order.model_dump()} | Output: {prediction_res} | Latency: {latency_ms:.2f}ms")

        return prediction_res
    except ValueError as ve:
        logger.warning(f"Validation failed: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict-batch", response_model=List[PredictionOutput], status_code=status.HTTP_200_OK)
def predict_batch(orders: List[OrderSchema]):
    try:
        start_time = datetime.utcnow()
        data = [order.model_dump() for order in orders]
        df = pd.DataFrame(data)
        
        # 1. طبقة التحقق من صحة البيانات
        validate_input_data(df)
        
        # 2. التنبؤ
        results = predictor.predict(df)

        # 3. تسجيل المراقبة (Prediction Logging)
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Batch Prediction Success | Count: {len(orders)} | Latency: {latency_ms:.2f}ms")

        return results
    except ValueError as ve:
        logger.warning(f"Batch Validation failed: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")