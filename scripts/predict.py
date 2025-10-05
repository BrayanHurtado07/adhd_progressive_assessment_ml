import os, joblib, psycopg2, pandas as pd
from dotenv import load_dotenv
from ml.data import read_features_labels

load_dotenv()
DSN = os.getenv("DATABASE_URL")
MODEL_DIR = os.getenv("MODEL_DIR","./artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_snap.joblib")

def features_for_student(student_id: str) -> pd.DataFrame:
    with psycopg2.connect(DSN) as con:
        X = pd.read_sql("SELECT * FROM adhd.vw_student_features WHERE student_id = %s", con, params=[student_id])
    return X

def main(student_id: str, model_id: str):
    model = joblib.load(MODEL_PATH)
    X = features_for_student(student_id)
    if X.empty: 
        raise SystemExit("Sin features para el estudiante.")
    pred = model.predict(X)[0]
    # confianza = prob. de la clase predicha
    proba = model.predict_proba(X)[0].max()

    with psycopg2.connect(DSN) as con:
        con.execute("""
            INSERT INTO adhd.ml_inferences(id,model_id,student_id,input_ref,prediction,severity,confidence)
            VALUES (gen_random_uuid(), %s, %s, NULL, %s, 'no_aplica', %s)
        """, (model_id, student_id, pred, float(proba)))
        con.commit()
    print("Predicción registrada:", pred, proba)

if __name__ == "__main__":
    import sys
    # python scripts/predict.py <student_id> <model_id>
    main(sys.argv[1], sys.argv[2])
