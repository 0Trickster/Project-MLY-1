import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficar_distribuciones_numericas(df: pd.DataFrame, bins: int = 30) -> None:
    """Histograma + KDE para cada columna numérica."""
    num_cols = df.select_dtypes(include=np.number).columns
    n = len(num_cols)
    if n == 0:
        return
    cols = 3
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4 * filas))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=axes[i])
        axes[i].set_title(f"Distribución de {col}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def graficar_boxplots(df: pd.DataFrame) -> None:
    """Boxplot para cada columna numérica (detección visual de outliers)."""
    num_cols = df.select_dtypes(include=np.number).columns
    n = len(num_cols)
    if n == 0:
        return
    cols = 3
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4 * filas))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f"Boxplot de {col}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def graficar_categoricas(df: pd.DataFrame, top_n: int = 10) -> None:
    """Countplot para cada columna categórica (top N categorías)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col == "track_id":
            continue
        top_cats = df[col].value_counts().head(top_n).index
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], order=top_cats, palette="viridis")
        plt.title(f"Top {top_n} en {col}")
        plt.tight_layout()
        plt.show()