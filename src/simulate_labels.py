# 📌 src/simulate_labels.py
import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("data/snap_dataset.csv")

# Crear etiquetas balanceadas
n_per_class = len(df) // 3
df["profile"] = ["inatento"] * n_per_class + ["hiperactivo"] * n_per_class + ["combinado"] * (len(df) - 2 * n_per_class)

# Asignar severidad balanceada
df["severity"] = ["leve", "moderado", "severo"] * (len(df) // 3) + ["leve"] * (len(df) % 3)

# Mezclar
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
df.to_csv("data/snap_dataset_labeled.csv", index=False)
print("✅ Dataset balanceado exportado.")