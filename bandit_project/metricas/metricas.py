import numpy as np
import pandas as pd
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

        best_arm_idx = np.argmax([b.classic_mean for b in brazos])
        best_arm_id = brazos[best_arm_idx].idBrazo

        return {
            "best_arm_idx": ("Índice Mejor Brazo:", best_arm_idx),
            "best_arm_id": ("ID Mejor Brazo:", best_arm_id)
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
        id_mejor = contexto["idMejorBrazo"][1]
        
        filtro = df['movieId'] == id_mejor
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
        likes = df_filtrado['recompensa'].sum()
        media = df_filtrado['recompensa'].mean()

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
        
        
        best_arm_id = contexto["idMejorBrazo"][1]
        historico = contexto["historicoRecomendados"]

        tail = historico[-self.window:]
        frecuencia = np.mean([any(b.idBrazo == best_arm_id for b in slate) for slate in tail])

        return {
            "frecuencia": ("Frecuencia de aparición del mejor brazo:", frecuencia)
        }


# ============================================================
# 4. ESTABILIDAD DEL SLATE
# ============================================================

class MetricaEstabilidadSlate(Metrica):
    nombre = "estabilidad_slate"

    def __init__(self, slate):
        super().__init__(slate=slate)
        self.slate = slate

    def calcular(self, contexto):
        historico = contexto["historicoRecomendados"]

        interseccion = []
        for i in range(1, len(historico)):
            slateA = historico[i]
            slateB = historico[i - 1]

            setA = {b.idBrazo for b in slateA}
            setB = {b.idBrazo for b in slateB}

            interseccion.append(len(setA & setB) / self.slate)

        slate_stability = np.mean(interseccion)

        return {
            "slate_stability": ("% Estabilidad del slate:", slate_stability * 100)
        }


# ============================================================
# 5. ENTROPÍA DE RECOMENDACIONES
# ============================================================

class MetricaEntropiaRecomendaciones(Metrica):
    nombre = "entropia_recomendaciones"

    def __init__(self):
        super().__init__()

    def calcular(self, contexto):
        brazos = contexto["brazos"]
        historico = contexto["historicoRecomendados"]

        conteo = Counter(b.idBrazo for slate in historico for b in slate)
        total = sum(conteo.values())

        entropia = -sum((c / total) * math.log(c / total) for c in conteo.values())
        num_brazos = len(brazos)

        return {
            "entropia": ("Entropía:", entropia),
            "numBrazos": ("Número de brazos:", num_brazos),
            "entropiaMaxima": ("Entropía máxima:", math.log(num_brazos)),
            "brazosusados": ("Número efectivo de brazos usados:", math.exp(entropia))
        }


# ============================================================
# 6. VARIACIÓN DE REWARDS POR BATCH
# ============================================================

class MetricaVariacionRewards(Metrica):
    nombre = "variacion_rewards"

    def __init__(self, window):
        super().__init__(window=window)
        self.window = window

    def calcular(self, contexto):
        sz_avg_reward_acum = contexto["sz_avg_reward_acum"]

        tail = sz_avg_reward_acum[-self.window:]
        variacion_rel_global = (tail.max() - tail.min()) / tail.mean()

        return {
            "ventana": ("Tamaño de ventana deslizante:", self.window),
            "variacion_rel_global": ("Variación relativa:", variacion_rel_global)
        }


# ============================================================
# 7. PRIMER BATCH DE CONVERGENCIA
# ============================================================

class MetricaPrimerBatchConvergencia(Metrica):
    nombre = "primer_batch_convergencia"

    def __init__(self, margen):
        super().__init__(margen=margen)
        self.margen = margen

    def calcular(self, contexto):
        sz_avg_reward_acum = contexto["sz_avg_reward_acum"]

        margen_rel = self.margen / 100
        cumavg = sz_avg_reward_acum

        stable = np.where(np.abs(cumavg - cumavg[-1]) < 0.02 * cumavg[-1])[0]
        t_convergencia = stable[0] if len(stable) > 0 else None

        return {
            "mediaAcumFinal": ("Media acumulada final:", sz_avg_reward_acum[-1]),
            "margen": ("Margen relativo:", margen_rel),
            "t_convergencia": ("Primer batch dentro del margen:", t_convergencia)
        }


# ============================================================
# 8. VARIACIÓN RELATIVA DE MEDIAS
# ============================================================

class MetricaVariacionRelativaMedias(Metrica):
    nombre = "variacion_relativa_medias"

    def __init__(self, window):
        super().__init__(window=window)
        self.window = window

    def calcular(self, contexto):
        historico = contexto["historicoMeansPorBatch"]

        matriz = np.vstack(historico)
        tail = matriz[-self.window:]

        margen = tail.max(axis=0) - tail.min(axis=0)
        media_final = matriz[-1]

        variaciones = margen / np.maximum(media_final, 1e-6)

        return {
            "variaciones_relativas": ("Vector de variaciones relativas:", variaciones),
            "VariacionMedia": ("Variación relativa media:", variaciones.mean()),
            "min_variacion": ("Máxima variación (peor brazo):", variaciones.max()),
            "max_variacion": ("Mínima variación (mejor brazo):", variaciones.min()),
            "p95_variacion": ("Percentil 95:", np.percentile(variaciones, 95))
        }

# ============================================================
# 8. Media efectiva del slat
# ============================================================
class MetricaMediaEfectivaSlate(Metrica):
    nombre = "media_efectiva_slate"

    def __init__(self):
        super().__init__()

    def calcular(self, contexto):
        mean_slate_history = np.array(contexto["mean_slate_history"])

        # ignorar NaN si los hay
        valid = ~np.isnan(mean_slate_history)
        if not valid.any():
            return {
                "media_final": ("Media efectiva final del slate:", None),
                "media_global": ("Media efectiva media del slate:", None),
            }

        media_final = mean_slate_history[valid][-1]
        media_global = mean_slate_history[valid].mean()

        return {
            "media_final": ("Media efectiva final del slate:", media_final),
            "media_global": ("Media efectiva media del slate:", media_global),
        }


class MetricaGapEfectivo(Metrica):
    nombre = "gap_efectivo"

    def __init__(self):
        super().__init__()

    def calcular(self, contexto):
        mean_slate_history = np.array(contexto["mean_slate_history"])
        best_arm_effective_history = np.array(contexto["best_arm_effective_history"])

        valid = ~np.isnan(mean_slate_history)
        if not valid.any():
            return {
                "gap_final": ("Gap efectivo final (best - slate):", None),
                "gap_medio": ("Gap efectivo medio:", None),
            }

        gap = best_arm_effective_history[valid] - mean_slate_history[valid]

        return {
            "gap_final": ("Gap efectivo final (best - slate):", gap[-1]),
            "gap_medio": ("Gap efectivo medio:", gap.mean()),
        }
