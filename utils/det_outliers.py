import pandas as pd
import numpy as np
from scipy import stats
def detectar_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta outliers usando el método del rango intercuartílico (IQR)."""
    num_df = df.select_dtypes(include=np.number)
    resultados = []
    for col in num_df.columns:
        q1, q3 = num_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((num_df[col] < lim_inf) | (num_df[col] > lim_sup)).sum()
        resultados.append({
            "columna": col, "limite_inferior": lim_inf, "limite_superior": lim_sup,
            "n_outliers": n_outliers, "porcentaje_%": round(n_outliers / len(df) * 100, 2)
        })
    tabla = pd.DataFrame(resultados).sort_values("n_outliers", ascending=False)
    print(tabla)
    return tabla


def detectar_outliers_zscore(df: pd.DataFrame, umbral: float = 3.0) -> pd.DataFrame:
    """Detecta outliers usando z-score."""
    num_df = df.select_dtypes(include=np.number)
    resultados = []
    for col in num_df.columns:
        z = np.abs(stats.zscore(num_df[col].dropna()))
        n_outliers = (z > umbral).sum()
        resultados.append({"columna": col, "n_outliers": n_outliers,
                            "porcentaje_%": round(n_outliers / len(df) * 100, 2)})
    tabla = pd.DataFrame(resultados).sort_values("n_outliers", ascending=False)
    print(tabla)
    return tabla