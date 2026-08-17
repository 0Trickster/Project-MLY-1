import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def matriz_correlacion(df: pd.DataFrame, metodo: str = "pearson") -> pd.DataFrame:
    """Heatmap de correlación entre variables numéricas."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        print("No hay suficientes columnas numéricas para correlación.")
        return pd.DataFrame()
    corr = num_df.corr(method=metodo)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"Matriz de correlación ({metodo})")
    plt.tight_layout()
    plt.show()
    return corr


def pares_alta_correlacion(df: pd.DataFrame, umbral: float = 0.8) -> pd.DataFrame:
    """Lista los pares de variables con |correlación| mayor al umbral."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().abs()
    pares = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    pares = pares[pares > umbral].sort_values(ascending=False)
    print(f"Pares con correlación > {umbral}:")
    print(pares if not pares.empty else "Ninguno.")
    return pares.reset_index().rename(columns={0: "correlacion"})


def pairplot(df: pd.DataFrame, columnas: list = None, hue: str = None) -> None:
    """Pairplot (dispersión + distribución) para un subconjunto de columnas."""
    cols = columnas if columnas else df.select_dtypes(include=np.number).columns[:5]
    sns.pairplot(df, vars=cols, hue=hue, diag_kind="kde")
    plt.show()


def scatter_vs_target(df: pd.DataFrame, target: str) -> None:
    """Scatterplot de cada variable numérica contra la variable objetivo."""
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()