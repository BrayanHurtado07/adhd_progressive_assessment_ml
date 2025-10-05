# seed_to_db.py
import os, psycopg2, pandas as pd, numpy as np
from dotenv import load_dotenv
from simulate_data import simulate
load_dotenv()
DSN = os.getenv("DATABASE_URL")

INSTITUTION_ID = None     # si ya tienes una, ponla aquí; si no, se crea.

def main(n=120):
    df = simulate(n=n)
    with psycopg2.connect(DSN, autocommit=False) as con:
        cur = con.cursor()
        # institución
        global INSTITUTION_ID
        if INSTITUTION_ID is None:
            cur.execute("INSERT INTO adhd.institutions(id,name,code) VALUES (gen_random_uuid(),'Demo Inst','DEMO') RETURNING id")
            INSTITUTION_ID = cur.fetchone()[0]

        for _, r in df.iterrows():
            # estudiante + PII
            cur.execute("""
              INSERT INTO adhd.students(id,institution_id,code) VALUES (gen_random_uuid(),%s,%s) RETURNING id
            """, (INSTITUTION_ID, r["student_id"]))
            student_id = cur.fetchone()[0]
            cur.execute("""
              INSERT INTO adhd.student_pii(student_id,full_name,birthdate,gender)
              VALUES (%s, %s, %s, %s)
            """, (student_id, f"Alumno {r['student_id']}", "2014-01-01", r["gender"]))

            # snap_form + scores (vamos directo, para simular)
            cur.execute("""
              INSERT INTO adhd.snap_forms(id,institution_id,student_id,created_at)
              VALUES (gen_random_uuid(), %s, %s, now())
              RETURNING id
            """, (INSTITUTION_ID, student_id))
            form_id = cur.fetchone()[0]
            cur.execute("""
              INSERT INTO adhd.snap_scores(form_id,inattentive_score,hyperimpulsive_score,odd_score,total_score,severity)
              VALUES (%s,%s,%s,%s,%s, CASE WHEN %s>=60 THEN 'severo'
                                           WHEN %s>=40 THEN 'moderado'
                                           WHEN %s>=20 THEN 'leve'
                                           ELSE 'no_aplica' END::adhd.severity_t)
            """, (form_id, int(r.inattentive_score), int(r.hyperimpulsive_score),
                  int(r.odd_score), int(r.total_score),
                  int(r.total_score), int(r.total_score), int(r.total_score)))

            # KPIs actividades: una sesión agregada para alimentar la vista
            cur.execute("""
              WITH a AS (
                INSERT INTO adhd.activities(id,code,title) VALUES (gen_random_uuid(),'DEMO','Demo') ON CONFLICT DO NOTHING
                RETURNING id
              )
              SELECT COALESCE((SELECT id FROM a), (SELECT id FROM adhd.activities WHERE code='DEMO')) 
            """)
            activity_id = cur.fetchone()[0]
            cur.execute("""
              INSERT INTO adhd.activity_sessions(id,student_id,activity_id,started_at,finished_at)
              VALUES (gen_random_uuid(), %s, %s, now() - interval '1 day', now())
              RETURNING id
            """, (student_id, activity_id))
            session_id = cur.fetchone()[0]
            cur.execute("""
              INSERT INTO adhd.activity_metrics(session_id,success_rate,response_time_ms,attempts,omissions,errors)
              VALUES (%s,%s,%s, 1, 0, 0)
            """, (session_id, float(r.avg_success_30d), int(r.avg_time_ms_30d)))

            # etiqueta 'reglas' para entrenamiento bootstrap
            cur.execute("""
              INSERT INTO adhd.cognitive_profiles(id,student_id,source,tdah,severity,generated_at)
              VALUES (gen_random_uuid(), %s, 'reglas', CASE WHEN %s=1 THEN 'combinado' ELSE 'no_clasificado' END::adhd.tdah_type, 'no_aplica', now())
            """, (student_id, int(r.label_adhd)))
        con.commit()
    print(f"Sembrados {n} estudiantes en la BD.")

if __name__ == "__main__":
    main(150)   # cambia el tamaño para “pocos/medianos”
