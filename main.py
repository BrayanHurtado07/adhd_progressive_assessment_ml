# 📌 api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from src.transform import transform_input

app = FastAPI()

# Modelos cargados
model_rf = joblib.load("models/modelo_perfil_rf.pkl")
model_mlp = joblib.load("models/modelo_perfil_mlp.pkl")
scaler = joblib.load("models/escalador.pkl")

# 🎯 Modelo por defecto
MODEL_TYPE = "rf"  # o "mlp"

# 🎯 Entrada esperada (respuestas 1 a 26)
class SnapInput(BaseModel):
    responses: List[int]  # 26 respuestas SNAP

@app.post("/predict-profile")
def predict_profile(data: SnapInput):
    if len(data.responses) != 26:
        raise HTTPException(status_code=400, detail="Se requieren exactamente 26 respuestas.")

    df = pd.DataFrame([data.responses], columns=[str(i) for i in range(1, 27)])

    if MODEL_TYPE == "mlp":
        X = transform_input(df, scale=True)
        prediction = model_mlp.predict(X)[0]
    else:
        X = transform_input(df, scale=False)
        prediction = model_rf.predict(X)[0]

    return {"predicted_profile": prediction}