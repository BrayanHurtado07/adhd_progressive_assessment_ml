# 📌 Ruta: adhd_ml_model/src/extract.py

import psycopg2
import pandas as pd

# 🔗 Conexión a tu PostgreSQL Railway
conn = psycopg2.connect(
    dbname="railway",
    user="postgres",
    password="ddwejfmuKeWOVPYtNCnyPztotPNbJyvi",
    host="mainline.proxy.rlwy.net",
    port="43200"
)

# 🧾 Consulta SQL: obtener todas las respuestas SNAP por formulario
query = """
SELECT
    sf.id AS form_id,
    sf.student_id,
    sa.question_id,
    so.value AS response_value
FROM snap_forms sf
JOIN snap_answers sa ON sf.id = sa.form_id
JOIN snap_options so ON sa.option_id = so.id
ORDER BY sf.id, sa.question_id;
"""

# 📥 Cargar a DataFrame
df = pd.read_sql_query(query, conn)
conn.close()

# 🔄 Transformar a formato ancho: 1 fila = 1 formulario con 26 columnas
df_wide = df.pivot(index='form_id', columns='question_id', values='response_value')
df_wide.reset_index(inplace=True)

# 💾 Guardar como CSV
df_wide.to_csv("data/snap_dataset.csv", index=False)
print("✅ Dataset exportado a: data/snap_dataset.csv")