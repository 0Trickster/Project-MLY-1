
import numpy as np
import pandas as pd
def estadisticas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos ampliados para columnas numéricas."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        print("No hay columnas numéricas.")
        return pd.DataFrame()
    desc = num_df.describe().T
    desc["skew"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()
    desc["rango"] = desc["max"] - desc["min"]
    return desc


def estadisticas_categoricas(df: pd.DataFrame) -> None:
    """Frecuencias y moda de columnas categóricas."""
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        print("No hay columnas categóricas.")
        return
    for col in cat_df.columns:
        #En caso de que se quiera saber la cancion que mas se repite quitar esto
        if col == "track_id":
            continue
        print(f"\n--- {col} ---")
        print(f"Moda: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
        print(df[col].value_counts(dropna=False).head(10))