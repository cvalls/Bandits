import numpy as np
import math
from collections import Counter

# ============================================================
# CLASE BASE
# ============================================================

class Metrica:
    """
    Clase base para todas las métricas del sistema.
    Cada métrica debe implementar el método calcular(contexto).
    """
    nombre = "base"

    def __init__(self, **kwargs):
        self.params = kwargs

    def calcular(self, contexto):
        raise NotImplementedError("Debe implementarse en la subclase.")


# ============================================================
# 1. MEJOR BRAZO FINAL
# ============================================================

class MetricaMejorBrazoFinal(Metrica):
    nombre = "mejor_brazo_final"

    def __init__(self):
        super().__init__()

    def calcular(self, contexto):
        brazos = contexto["brazos"]

        best_arm_idx = np.argmax([b.means for b in brazos])
        best_arm_id = brazos[best_arm_idx].idBrazo

        return {
            "best_arm_idx": ("Índice del mejor brazo:", best_arm_idx),
            "best_arm_id": ("ID del mejor brazo:", best_arm_id)
        }


# ============================================================
# 2. DATOS DEL MEJOR BRAZO
# ============================================================

class MetricaDatosMejorPeli(Metrica):
    nombre = "datos_mejor_peli"

    def __init__(self):
        super().__init__()

    def calcular(self, contexto):
        df = contexto["df"]
        id_mejor_brazo = contexto["idMejorBrazo"]

        filtro = df['movieId'] == id_mejor_brazo
        df_filtrado = df.loc[filtro]
        
        if df_filtrado.empty:
            return {
                "titulo": ("Título no encontrado", None),
                "veces": ("Número de veces presente en el log", 0),
                "likes": ("Número de likes en el log", 0),
                "media": ("Media en el log", None)
            }
        
        titulo = df_filtrado['title'].iloc[0]
        veces = filtro.sum()
        likes = df_filtrado['liked'].sum()
        media = df_filtrado['liked'].mean()
        
        return {
            "titulo": ("Título", titulo),
            "veces": ("Número de veces presente en el log", veces),
            "likes": ("Número de likes en el log", likes),
            "media": ("Media en el log", media)
        }


# ============================================================
# 3. FRECUENCIA DEL MEJOR BRAZO
# ============================================================

class MetricaFrecuenciaMejorBrazo(Metrica):
    nombre = "frecuencia_mejor_brazo"

    def __init__(self, window):
        super().__init__(window=window)
        self.window = window

    def calcular(self, contexto):
        best_arm_id = contexto["idMejorBrazo"]
        historico_recs = contexto["historicoRecomendados"]

