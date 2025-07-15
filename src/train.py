# 📌 src/train.py

import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.transform import transform_input

def train_models():
    # Cargar datos
    df = pd.read_csv("data/snap_dataset_labeled.csv")
    X = df[[str(i) for i in range(1, 27)]]
    y = df["profile"]

    # Escalado para MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/escalador.pkl")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "models/modelo_perfil_rf.pkl")

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X_scaled, y)
    joblib.dump(mlp, "models/modelo_perfil_mlp.pkl")

    print("✅ Modelos entrenados y guardados.")

    # Validación cruzada rápida (opcional)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_score = cross_val_score(rf, X, y, cv=cv).mean()
    mlp_score = cross_val_score(mlp, X_scaled, y, cv=cv).mean()
    print(f"📊 CV Accuracy RF: {rf_score:.2f} | MLP: {mlp_score:.2f}")

if __name__ == "__main__":
    train_models()