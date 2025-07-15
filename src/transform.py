# 📌 src/transform.py

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Ruta del modelo de escalado
SCALER_PATH = "models/escalador.pkl"

def transform_input(data: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
    """
    Transforma los datos de entrada:
    - Selecciona columnas 1 a 26 (respuestas SNAP).
    - Aplica escalado si se desea.

    Args:
        data (pd.DataFrame): Datos de entrada.
        scale (bool): Si se debe aplicar escalado.

    Returns:
        pd.DataFrame: Datos procesados (escalados si aplica).
    """
    X = data[[str(i) for i in range(1, 27)]]

    if scale:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"⚠️ Scaler no encontrado en: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
        return X_scaled

    return X