import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def simulate(n=1200, pos_rate=0.35, label_noise=0.10):
    """
    Dataset con solape entre clases y ruido de etiqueta.
    y = 1 (ADHD), 0 (no-ADHD)
    """
    y = (rng.random(n) < pos_rate).astype(int)

    # SNAP con solape (medias más cercanas + sigma mayor)
    inatt = np.clip(rng.normal(11 + 6*y, 5.5, n), 0, 27)
    hyper = np.clip(rng.normal(10 + 5*y, 5.0, n), 0, 27)
    odd   = np.clip(rng.normal( 4 + 2*y, 4.0, n), 0, 24)

    # ruido discreto tipo medición
    inatt = np.round(inatt + rng.normal(0, 0.6, n))
    hyper = np.round(hyper + rng.normal(0, 0.6, n))
    odd   = np.round(odd   + rng.normal(0, 0.6, n))

    # KPIs actividades (diferencias suaves)
    sessions    = np.clip(rng.poisson(9 - 1.2*y), 1, 20)
    avg_success = np.clip(rng.normal(0.84 - 0.10*y, 0.09, n), 0.40, 0.98)  # 0..1
    avg_time    = np.clip(rng.normal(1150 + 180*y, 240, n), 600, 2600)     # ms

    # Demografía
    age    = rng.integers(7, 13, n)
    gender = rng.choice(["M", "F"], size=n, p=[0.6, 0.4])

    # Ruido de etiqueta
    flip = rng.random(n) < label_noise
    y_noisy = y.copy()
    y_noisy[flip] = 1 - y_noisy[flip]

    total = inatt + hyper + odd  # para inspección (NO usar como feature)

    df = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "inattentive_score": inatt.astype(int),
        "hyperimpulsive_score": hyper.astype(int),
        "odd_score": odd.astype(int),
        "total_score": total.astype(int),          # no se usa como feature
        "sessions_30d": sessions.astype(int),
        "avg_success_30d": (avg_success * 100).round(2),  # %
        "avg_time_ms_30d": avg_time.round(0).astype(int),
        "label_adhd": y_noisy.astype(int)
    })
    return df

if __name__ == "__main__":
    print(simulate(5))
