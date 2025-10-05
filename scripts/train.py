import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from ml.data import read_features_labels
from ml.reweigh import reweigh_by_group
import psycopg

load_dotenv()
DSN = os.getenv("DATABASE_URL")
MODEL_DIR = os.getenv("MODEL_DIR", "./artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_xy(df: pd.DataFrame):
    y = df["tdah"].fillna("no_clasificado")
    X = df.drop(columns=["student_id","tdah","severity"])
    num_cols = ["age","inattentive_score","hyperimpulsive_score","odd_score","total_score",
                "sessions_30d","avg_success_30d","avg_time_ms_30d"]
    cat_cols = ["gender"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="drop")
    return X, y, pre

def main():
    df = read_features_labels()
    if df.empty: 
        raise SystemExit("No hay datos etiquetados para entrenar.")

    X, y, pre = build_xy(df)

    # pesos por equidad (grupo = gender)
    w = reweigh_by_group(pd.concat([X[["gender"]], y], axis=1), "gender", "tdah")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=y
    )

    # 1) Regresión Logística
    logreg = Pipeline([("pre", pre),
                       ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))])
    logreg.fit(X_train, y_train, clf__sample_weight=w_train)

    # 2) Random Forest
    rf = Pipeline([("pre", pre),
                   ("clf", RandomForestClassifier(
                        n_estimators=300, max_depth=None, class_weight="balanced_subsample", random_state=42))])
    rf.fit(X_train, y_train, clf__sample_weight=w_train)

    def eval_model(name, model):
        proba = model.predict_proba(X_test)
        # multi-clase: macro AUC (promedio de one-vs-rest)
        aucs = []
        for i, cls in enumerate(model.named_steps["clf"].classes_):
            aucs.append(roc_auc_score((y_test==cls).astype(int), proba[:, i]))
        print(f"{name} AUC macro: {sum(aucs)/len(aucs):.3f}")
        print(classification_report(y_test, model.predict(X_test)))

    eval_model("LogReg", logreg)
    eval_model("RandomForest", rf)

    # Guarda el mejor (comienza con RF)
    model_path = os.path.join(MODEL_DIR, "rf_snap.joblib")
    joblib.dump(rf, model_path)
    print("Guardado:", model_path)

    # Registra el modelo en la tabla ml_models
    with psycopg.connect(DSN) as con:
        cur = con.execute(
            "INSERT INTO adhd.ml_models (id,name,version,metrics) VALUES (gen_random_uuid(),%s,%s,%s) RETURNING id",
            ("rf-snapiv","1.0.0", {"note":"AUC macro en consola"}))
        model_id = cur.fetchone()[0]
        con.commit()
        print("Registrado en ml_models:", model_id)

if __name__ == "__main__":
    main()
