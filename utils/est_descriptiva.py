import numpy as np
import pandas as pd


def estadisticas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticos descriptivos ampliados para las columnas numéricas.

    Además de los estadísticos básicos proporcionados por `describe()`,
    calcula la asimetría (skewness), la curtosis y el rango de cada variable.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables a analizar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los estadísticos descriptivos de cada variable numérica.
        Si no existen columnas numéricas, retorna un DataFrame vacío.
    """

    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print('No hay columnas numéricas.')
        return pd.DataFrame()

    desc = num_df.describe().T

    desc['skew'] = num_df.skew()
    desc['kurtosis'] = num_df.kurtosis()
    desc['rango'] = desc['max'] - desc['min']

    return desc


def estadisticas_categoricas(df: pd.DataFrame) -> None:
    """
    Calcula y muestra estadísticas descriptivas de las variables categóricas.

    Para cada columna categórica se muestra su moda y las diez categorías
    con mayor frecuencia, incluyendo los valores nulos en el conteo.

    La columna `track_id` se excluye del análisis debido a que corresponde
    a un identificador y su frecuencia no resulta relevante para este análisis.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables categóricas a analizar.

    Retorna
    -------
    None
        La función imprime los resultados directamente en consola.
    """

    cat_df = df.select_dtypes(include=['object', 'category'])

    if cat_df.empty:
        print('No hay columnas categóricas.')
        return

    for col in cat_df.columns:

        # Se excluye el identificador del análisis.
        if col == 'track_id':
            continue

        print(f'\n--- {col} ---')

        moda = df[col].mode()

        print(
            f"Moda: {moda.iloc[0] if not moda.empty else 'N/A'}"
        )

        print(
            df[col].value_counts(dropna=False).head(10)
        )


def sturges_bins(data) -> int:
    """
    Calcula el número de bins recomendado mediante la regla de Sturges.

    La regla de Sturges estima el número de intervalos de un histograma
    en función de la cantidad de observaciones disponibles.

    Parámetros
    ----------
    data : array-like
        Conjunto de datos unidimensionales utilizado para calcular
        el número de observaciones.

    Retorna
    -------
    int
        Número de bins recomendado para el histograma.

    Raises
    ------
    ValueError
        Si el conjunto de datos está vacío.
    """

    n = len(data)

    if n == 0:
        raise ValueError('El conjunto de datos está vacío.')

    return int(np.log2(n)) + 1


def freedman_diaconis_bins(
    df: pd.DataFrame,
    column: str
) -> int:
    """
    Calcula el número de bins óptimos usando la regla de Freedman-Diaconis.

    La regla de Freedman-Diaconis utiliza el rango intercuartílico (IQR)
    y la cantidad de observaciones para determinar el ancho de los
    intervalos de un histograma.

    Los valores no numéricos y los valores nulos de la columna son
    convertidos o eliminados antes del cálculo.

    Si el IQR es igual a cero, se utiliza una aproximación basada en
    la desviación estándar como alternativa. Si tanto el IQR como la
    desviación estándar son cero, se devuelve un único bin.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos.
    column : str
        Nombre de la columna numérica que se analizará.

    Retorna
    -------
    int
        Número de bins recomendado para el histograma.

    Raises
    ------
    ValueError
        Si la columna contiene menos de dos valores numéricos válidos.
    KeyError
        Si la columna especificada no existe en el DataFrame.
    """

    # Obtener la columna y convertir sus valores a numéricos.
    data = (
        pd.to_numeric(
            df[column],
            errors='coerce'
        )
        .dropna()
        .to_numpy()
    )

    # Verificar que existan suficientes observaciones.
    if len(data) < 2:
        raise ValueError(
            f"La columna '{column}' no contiene suficientes datos."
        )

    # Cantidad de observaciones.
    n = len(data)

    # Calcular el rango intercuartílico (IQR).
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Fallback cuando el IQR es igual a cero.
    if iqr == 0:
        std = np.std(data)

        if std == 0:
            return 1

        # Aproximación del IQR para una distribución normal.
        iqr = 1.349 * std

    # Calcular el ancho de cada bin.
    bin_width = 2 * iqr / (n ** (1 / 3))

    # Calcular el rango total de los datos.
    data_range = np.max(data) - np.min(data)

    if data_range == 0:
        return 1

    # Calcular el número de bins.
    n_bins = int(np.ceil(data_range / bin_width))

    return max(n_bins, 1)