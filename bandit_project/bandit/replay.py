import time
import numpy as np
import pandas as pd
import bandit.brazos as cs  # donde está crear_Brazo()
from typing import Dict, List, Optional

# NOTA SOBRE EL MÉTODO REPLAY:
# En evaluación offline, el tiempo del dataset histórico NO coincide con el tiempo interno del bandido.
# Replay no simula un bandido online real, sino que comprueba si las películas recomendadas aparecen
# en el log histórico. Si aparecen, asumimos que el usuario habría hecho clic ahora; si no, simplemente
# no hay información. Esto es normal y está asumido: el bandido aprende a partir de coincidencias en el
# dataset, no a partir de eventos generados en tiempo real. El orden temporal del dataset no se usa.

# filtra el conjunto de pelis inicial, con la lista de recomendadas
# y devuelve un array de parejas movieId y el valor del liked que tenía en el conjunto inicial
# así se considera ese like como si fuese la respuesta de la peli al evento.

# index_replay es un diccionario clave movie_id y valor array de posiciones en el df
# recs es una lista de objetos Brazo
# la salida es un dataframe con las filas emparejadas


def filtrar_eventos_replay(
    index_replay: Dict[int, List[int]],
    df: pd.DataFrame,
    recs: List[cs.Brazo],
    t: int,
    batch_size: int
) -> Optional[pd.DataFrame]:

    start = t                      # inicio del batch
    end = t + batch_size           # fin del batch (no incluido)
    filas_en_batch = []            # renombre interno seguro

    # para cada peli recomendada
    for brazo in recs:
        movie_id = brazo.idBrazo   # renombre interno seguro

        if movie_id in index_replay:
            posiciones_peli = index_replay[movie_id]  # renombre interno seguro

            # añadir solo las posiciones dentro del batch actual
            filas_en_batch.extend(
                [i for i in posiciones_peli if start <= i < end]
            )

    # si no hay filas, no hay emparejamientos
    if not filas_en_batch:
        return None

    # extraer movieId y liked de las filas emparejadas
    acciones = df.loc[filas_en_batch, ["movieId", "recompensa", "recompensa_cat"]].copy()
    acciones["t"] = t

    return acciones


