import numpy as np
import pandas as pd

def reweigh_by_group(df: pd.DataFrame, group_col: str, label_col: str):
    # Kamiran & Calders (versión simple): balancea P(Y|A)
    w = np.ones(len(df), dtype=float)
    for g, gdf in df.groupby(group_col, dropna=False):
        for y, gy in gdf.groupby(label_col):
            p_overall = (df[label_col]==y).mean()
            p_group   = (gdf[label_col]==y).mean() if len(gdf) else 1.0
            if p_group and p_overall:
                w[gy.index] = p_overall / p_group
    return w
