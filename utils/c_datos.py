import pandas as pd
def valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de valores nulos por columna."""

    nulos = df.isnull().sum()
    porcentaje = (nulos / len(df)) * 100

    tabla = pd.DataFrame({
        "nulos": nulos,
        "porcentaje_%": porcentaje
    })

    tabla = tabla[
        tabla["nulos"] > 0
    ].sort_values("porcentaje_%", ascending=False)

    print("Valores nulos por columna:")
    print(tabla if not tabla.empty else "No hay valores nulos.")

    return tabla

def duplicados(df: pd.DataFrame) -> int:
    """Cuenta y muestra filas duplicadas."""
    n_dup = df.duplicated().sum()
    print(f"Filas duplicadas: {n_dup} ({n_dup/len(df)*100:.2f}%)")
    if n_dup > 0:
        print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head(10))
    return n_dup

def valores_unicos(df: pd.DataFrame) -> pd.DataFrame:
    """Cardinalidad (nº de valores únicos) por columna."""
    tabla = pd.DataFrame({
        "valores_unicos": df.nunique(),
        "porcentaje_%": (df.nunique() / len(df) * 100).round(2)
    }).sort_values("valores_unicos", ascending=False)
    print(tabla)
    return tabla