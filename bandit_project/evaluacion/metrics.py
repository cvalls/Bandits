import numpy as np

def moving_average(series, window=1000):
    """
    Calcula la media mÃ³vil de una serie numÃ©rica.

    ParÃ¡metros
    ----------
    series : list or np.ndarray
        Serie de valores numÃ©ricos.
    window : int
        TamaÃ±o de la ventana de suavizado.

    Retorna
    -------
    np.ndarray
        Serie suavizada mediante media mÃ³vil.
    """
    if len(series) < window:
        return np.array(series)

    cumsum = np.cumsum(np.insert(series, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)
