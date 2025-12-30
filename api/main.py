from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn

# Crear aplicación FastAPI
app = FastAPI(
    title="Online Shoppers Purchasing Intention API",
    description="API REST para predecir la intención de compra de visitantes en e-commerce",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rutas de modelos (compatible con ejecución local y Docker)
# Priorizar modelo comprimido (optimizado) sobre original
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = '/app/models'

MODEL_PATH_COMPRESSED = os.path.join(MODELS_DIR, 'best_model_compressed.pkl')
MODEL_PATH_ORIGINAL = os.path.join(MODELS_DIR, 'best_model.pkl')
MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'model_info.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Cargar modelo (intentar comprimido primero)
try:
    if os.path.exists(MODEL_PATH_COMPRESSED):
        print(f"📂 Cargando modelo comprimido (optimizado memoria)")
        model = joblib.load(MODEL_PATH_COMPRESSED)
    else:
        print(f"📂 Cargando modelo original")
        model = joblib.load(MODEL_PATH_ORIGINAL)
    
    # Cargar scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Cargar información
    try:
        model_info = joblib.load(MODEL_INFO_PATH)
    except:
        model_info = {'model_name': 'Gradient Boosting Classifier', 'n_features': 24}
    
    print(f"✅ Modelo cargado: {model_info.get('model_name', 'Gradient Boosting')}")
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo cargar el modelo: {e}")
    model = None
    model_info = None
    scaler = None

# Esquema de entrada para predicciones
class ShopperData(BaseModel):
    Administrative: int = Field(..., ge=0, description="Número de páginas administrativas visitadas")
    Administrative_Duration: float = Field(..., ge=0, description="Tiempo en páginas administrativas (segundos)")
    Informational: int = Field(..., ge=0, description="Número de páginas informativas visitadas")
    Informational_Duration: float = Field(..., ge=0, description="Tiempo en páginas informativas (segundos)")
    ProductRelated: int = Field(..., ge=0, description="Número de páginas de productos visitadas")
    ProductRelated_Duration: float = Field(..., ge=0, description="Tiempo en páginas de productos (segundos)")
    BounceRates: float = Field(..., ge=0, le=1, description="Tasa de rebote promedio")
    ExitRates: float = Field(..., ge=0, le=1, description="Tasa de salida promedio")
    PageValues: float = Field(..., ge=0, description="Valor promedio de páginas")
    SpecialDay: float = Field(..., ge=0, le=1, description="Proximidad a fecha especial")
    Month: str = Field(..., description="Mes de la sesión (Jan-Dec)")
    OperatingSystems: int = Field(..., ge=1, le=8, description="Sistema operativo (1-8)")
    Browser: int = Field(..., ge=1, le=13, description="Navegador (1-13)")
    Region: int = Field(..., ge=1, le=9, description="Región geográfica (1-9)")
    TrafficType: int = Field(..., ge=1, le=20, description="Tipo de tráfico (1-20)")
    VisitorType: str = Field(..., description="Tipo de visitante (Returning_Visitor, New_Visitor, Other)")
    Weekend: bool = Field(..., description="¿Es fin de semana?")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 5,
                "ProductRelated_Duration": 120.5,
                "BounceRates": 0.02,
                "ExitRates": 0.05,
                "PageValues": 10.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False
            }
        }
    }

# Esquema de respuesta
class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    confidence: str
    recommendation: str

@app.get("/", tags=["General"])
def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Online Shoppers Purchasing Intention API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["General"])
def health_check():
    """Verificar estado de la API y modelo"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info_loaded": model_info is not None
    }

@app.get("/model-info", tags=["General"])
def get_model_info():
    """Obtener información del modelo"""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Información del modelo no disponible")
    
    return {
        "model_name": model_info.get('model_name', 'Gradient Boosting Classifier'),
        "training_date": model_info.get('training_date', 'N/A'),
        "n_features": model_info.get('n_features', 24),
        "train_samples": model_info.get('train_samples', 0),
        "test_samples": model_info.get('test_samples', 0),
        "metrics": {
            "accuracy": round(model_info.get('accuracy', 0), 4),
            "f1_score": round(model_info.get('f1_score', 0), 4),
            "roc_auc": round(model_info.get('roc_auc', 0), 4)
        }
    }

def preprocess_input(data: ShopperData) -> np.ndarray:
    """Preprocesar datos de entrada para predicción (debe coincidir con preprocesamiento de entrenamiento)"""
    
    # One-hot encoding para Month (10 categorías)
    months = ['Aug', 'Dec', 'Feb', 'Jul', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep']
    month_features = [1 if data.Month == m else 0 for m in months]
    
    # One-hot encoding para VisitorType (3 categorías)
    visitor_types = ['New_Visitor', 'Other', 'Returning_Visitor']
    visitor_features = [1 if data.VisitorType == vt else 0 for vt in visitor_types]
    
    # Weekend (binario)
    weekend_feature = 1 if data.Weekend else 0
    
    # Crear array con las 24 features en el orden correcto
    features = np.array([[
        data.Administrative, 
        data.Administrative_Duration,
        data.Informational, 
        data.Informational_Duration,
        data.ProductRelated, 
        data.ProductRelated_Duration,
        data.BounceRates, 
        data.ExitRates, 
        data.PageValues, 
        data.SpecialDay,
        *month_features,      # 10 features
        *visitor_features,    # 3 features
        weekend_feature       # 1 feature
    ]])
    
    return features

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: ShopperData):
    """
    Predecir intención de compra para un visitante individual
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Preprocesar datos
        features = preprocess_input(data)
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Determinar confianza
        if probability >= 0.7:
            confidence = "Alta"
        elif probability >= 0.4:
            confidence = "Media"
        else:
            confidence = "Baja"
        
        # Generar recomendación
        if prediction == 1:
            if probability >= 0.7:
                recommendation = "Usuario con alta probabilidad de compra. Mantener experiencia fluida."
            else:
                recommendation = "Usuario probable comprador. Considerar incentivos o remarketing."
        else:
            if probability >= 0.3:
                recommendation = "Usuario indeciso. Ofrecer ayuda o cupones de descuento."
            else:
                recommendation = "Usuario con baja intención. Capturar información para remarketing."
        
        return PredictionResponse(
            prediction=bool(prediction),
            probability=round(float(probability), 4),
            confidence=confidence,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(data_list: List[ShopperData]):
    """
    Predecir intención de compra para múltiples visitantes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        results = []
        for data in data_list:
            features = preprocess_input(data)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            results.append({
                "prediction": bool(prediction),
                "probability": round(float(probability), 4)
            })
        
        return {
            "predictions": results,
            "total": len(results),
            "positive_predictions": sum(1 for r in results if r["prediction"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

# Ejecutar servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
