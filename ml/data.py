import pandas as pd, psycopg
from dotenv import load_dotenv
import os
load_dotenv()

DSN = os.getenv("DATABASE_URL")

def read_features_labels():
    with psycopg.connect(DSN) as con:
        X = pd.read_sql("SELECT * FROM adhd.vw_student_features", con)
        y = pd.read_sql("SELECT student_id, tdah FROM adhd.vw_training_labels", con)
    df = X.merge(y, on="student_id", how="inner")
    return df
