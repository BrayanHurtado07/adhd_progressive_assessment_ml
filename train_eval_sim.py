import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from simulate_data import simulate

def build_xy(df: pd.DataFrame):
    y = df["label_adhd"].astype(int)
    # OJO: excluimos total_score para no sobreinformar al modelo
    X = df.drop(columns=["student_id", "label_adhd", "total_score"])
    num = [
        "age", "inattentive_score", "hyperimpulsive_score", "odd_score",
        "sessions_30d", "avg_success_30d", "avg_time_ms_30d"
    ]
    cat = ["gender"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre

def pick_threshold_for_precision(y_true, proba, target_prec=0.98):
    """
    Elige el UMBRAL que maximiza RECALL sujeto a precisión >= objetivo.
    Si no existe, cae al umbral con mejor F1.
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)  # prec/rec: len n+1, thr: len n
    thr_pad = np.r_[thr, 1.01]  # alinear tamaños

    mask = prec >= target_prec
    if mask.any():
        # índice con mayor recall entre los que cumplen la precisión
        idxs = np.where(mask)[0]
        # evitar elegir el último punto (rec=0, umbral 1.01)
        best_idx = idxs[np.argmax(rec[idxs])]
        # si aún así cae en el sentinel, retrocede uno si es posible
        if best_idx == len(thr_pad) - 1 and len(idxs) > 1:
            best_idx = idxs[-2]
        return float(thr_pad[best_idx])

    # fallback: mejor F1
    eps = 1e-9
    f1 = 2 * (prec * rec) / (prec + rec + eps)
    best_idx = int(np.argmax(f1))
    return float(thr_pad[best_idx])

def eval_model(model, Xte, yte, target_prec=None, fixed_thr=None):
    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    if fixed_thr is not None:
        thr = fixed_thr
    elif target_prec is not None:
        thr = pick_threshold_for_precision(yte, proba, target_prec)
    else:
        thr = 0.5
    pred = (proba >= thr).astype(int)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    return auc, prec, rec, thr

def run(n=1200, seed=42):
    df = simulate(n=n, pos_rate=0.35, label_noise=0.10)
    X, y, pre = build_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # 1) Regresión Logística
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    logreg.fit(Xtr, ytr)
    auc_lr, p_lr, r_lr, thr_lr = eval_model(logreg, Xte, yte, target_prec=0.96)

    # 2) Random Forest
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=seed,
            class_weight="balanced_subsample"
        ))
    ])
    rf.fit(Xtr, ytr)
    auc_rf, p_rf, r_rf, thr_rf = eval_model(rf, Xte, yte, target_prec=0.98)

    # 3) SVM con probabilidades
    svm = Pipeline([
        ("pre", pre),
        ("clf", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale"))
    ])
    svm.fit(Xtr, ytr)
    auc_svm, p_svm, r_svm, thr_svm = eval_model(svm, Xte, yte, target_prec=0.92)

    print("\n=== Métricas (test) ===")
    print(f"Random Forest     AUC={auc_rf:.2f}  Precisión={p_rf:.2f}  Recall={r_rf:.2f}  (thr={thr_rf:.2f})")
    print(f"Regresión LogReg  AUC={auc_lr:.2f}  Precisión={p_lr:.2f}  Recall={r_lr:.2f}  (thr={thr_lr:.2f})")
    print(f"SVM               AUC={auc_svm:.2f} Precisión={p_svm:.2f} Recall={r_svm:.2f} (thr={thr_svm:.2f})")

    # Guarda el modelo RF
    joblib.dump(rf, "rf_snap_sim.joblib")
    print("\nGuardado: rf_snap_sim.joblib")

if __name__ == "__main__":
    run(n=1200, seed=42)  # n=600 (pocos), n=1200–2000 (medianos)
