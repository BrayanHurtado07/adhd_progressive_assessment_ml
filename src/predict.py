# 📌 src/predict.py

import pandas as pd
import joblib
from src.transform import transform_input

# Rutas de los modelos
MODEL_PATH_RF = "models/modelo_perfil_rf.pkl"
MODEL_PATH_MLP = "models/modelo_perfil_mlp.pkl"

def predict_profile(input_csv: str, model_type: str = "rf"):
    """
    Predice el perfil TDAH usando el modelo seleccionado.

    Args:
        input_csv (str): Ruta del archivo CSV con columnas 1 a 26 (respuestas).
        model_type (str): "rf" para Random Forest, "mlp" para Red Neuronal.

    Returns:
        pd.DataFrame: Resultados con predicciones.
    """
    # Cargar datos
    df = pd.read_csv(input_csv)

    # Preprocesamiento
    scale = model_type == "mlp"
    X = transform_input(df, scale=scale)

    # Cargar modelo
    model_path = MODEL_PATH_MLP if model_type == "mlp" else MODEL_PATH_RF
    model = joblib.load(model_path)

    # Predecir
    predictions = model.predict(X)
    result_df = df.copy()
    result_df["predicted_profile"] = predictions

    print("✅ Predicción completada.")
    return result_df

# 🧪 Ejemplo de uso
if __name__ == "__main__":
    resultado = predict_profile("data/snap_dataset.csv", model_type="rf")
    print(resultado[["predicted_profile"]].value_counts())