#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter


class BanditDashboard:

    def __init__(self, resultado):
        """
        Recibe un ResultadoExperimento y extrae de él TODO lo necesario.
        """

        # -----------------------------
        # Datos del contexto dashboard
        # -----------------------------
        ctx = resultado.contexto_dashboard

        self.resultado = resultado
        self.sz_num_emparejados_batch = np.array(ctx["sz_num_emparejados_batch"])
        self.history_list = ctx["history_list"]
        self.rewards = np.array(ctx["rewards"])
        self.sz_rewards_acumulado = np.array(ctx["sz_rewards_acumulado"])
        self.sz_avg_reward_acum = np.array(ctx["sz_avg_reward_acum"])
        self.ucb_mean_history = ctx["ucb_mean_history"]
        self.historicoMeansPorBatch = ctx["historicoMeansPorBatch"]
        self.window_variacion = ctx["window_variacion"]

        # -----------------------------
        # Datos del contexto diagnóstico
        # -----------------------------
        self.df = resultado.contexto_diagnostico["df"]

        # -----------------------------
        # Precalcular variaciones relativas
        # -----------------------------
        self._compute_variaciones_relativas()

    # ============================================================
    # CÁLCULO DE VARIACIÓN RELATIVA
    # ============================================================
    def _compute_variaciones_relativas(self):
        matriz_means = np.vstack(self.historicoMeansPorBatch)
        ventana_final = matriz_means[-self.window_variacion:]
        margen = ventana_final.max(axis=0) - ventana_final.min(axis=0)
        media_final = matriz_means[-1]

        self.variaciones_relativas = margen / np.maximum(media_final, 1e-6)
        self.tail = ventana_final
        self.mediaFinal = media_final

    # ============================================================
    # BLOQUE A – REPLAY / EVIDENCIA
    # ============================================================
    def plot_emparejamientos_por_batch(self, ax):
        ax.plot(self.sz_num_emparejados_batch, alpha=0.7, color='black')
        ax.set_title("Emparejamientos por batch")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Emparejamientos")
        ax.grid(True)

    def plot_emparejamientos_suavizados_por_batch(self, ax, window=200):
        rolling = np.convolve(self.sz_num_emparejados_batch, np.ones(window)/window, mode='valid')
        ax.plot(rolling, alpha=0.7, color='black')
        ax.set_title(f"Emparejamientos por batch suavizado (ventana {window})")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Emparejamientos")
        ax.grid(True)

    def plot_evidencia_acumulada(self, ax):
        total = self.sz_num_emparejados_batch.cumsum()
        ax.plot(total, color='black')
        ax.set_title("Evidencia acumulada")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Emparejamientos acumulados")
        ax.grid(True)

    def plot_k_por_pelicula(self, ax):
        k = self.df["movieId"].value_counts()
        ax.hist(k, bins=50, color='gray')
        ax.set_title("Distribución de k (apariciones en el log)")
        ax.set_xlabel("k")
        ax.set_ylabel("Número de películas")
        ax.grid(True)

    def plot_emparejamientos_por_pelicula(self, ax):
        pelis = [mid for (mid, recompensa, t) in self.history_list]
        conteo = Counter(pelis)
        serie = pd.Series(conteo)
        ax.hist(serie, bins=50, color='gray')
        ax.set_title("Emparejamientos por película (k·n real)")
        ax.set_xlabel("Emparejamientos")
        ax.set_ylabel("Número de películas")
        ax.grid(True)

    # ============================================================
    # BLOQUE B – REWARD GLOBAL
    # ============================================================
    def plot_reward_ruidoso(self, ax):
        ax.plot(self.rewards, color='red', alpha=0.4)
        ax.set_title("Reward por batch (ruidoso)")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward")
        ax.grid(True)

    def plot_reward_suavizado(self, ax, window=200):
        rolling = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
        ax.plot(rolling, color='red')
        ax.set_title(f"Reward suavizado (ventana {window})")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward suavizado")
        ax.grid(True)

    def plot_reward_acumulado(self, ax):
        ax.plot(self.sz_rewards_acumulado, color='red')
        ax.set_title("Reward acumulado")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward acumulado")
        ax.grid(True)

    def plot_media_acumulada(self, ax):
        ax.plot(self.sz_avg_reward_acum, color='red')
        ax.set_title("Media acumulada del reward")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward promedio")
        ax.grid(True)

    def plot_gradiente(self, ax):
        grad = np.gradient(self.sz_avg_reward_acum)
        ax.plot(grad, color='red')
        ax.set_title("Derivada de la media acumulada")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Δ Reward")
        ax.grid(True)

    # ============================================================
    # BLOQUE C – UCB
    # ============================================================
    def plot_ucb_mean(self, ax):
        ax.plot(self.ucb_mean_history, color='blue')
        ax.set_title("Evolución de la media de UCB")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Media UCB")
        ax.grid(True)

    # ============================================================
    # BLOQUE D – VARIACIÓN RELATIVA
    # ============================================================
    def plot_variacion_relativa_hist(self, ax):
        ax.hist(self.variaciones_relativas, bins=50, color='red', edgecolor='black')
        ax.set_title("Variación relativa (histograma)")
        ax.set_xlabel("Variación relativa")
        ax.set_ylabel("Número de brazos")
        ax.grid(True)

    def plot_variacion_relativa_boxplot(self, ax):
        ax.boxplot(self.variaciones_relativas, vert=False)
        ax.set_title("Boxplot de variación relativa")
        ax.set_xlabel("Variación relativa")
        ax.grid(True)

    def plot_variacion_relativa_ecdf(self, ax):
        vals = np.sort(self.variaciones_relativas)
        y = np.arange(1, len(vals)+1) / len(vals)
        ax.plot(vals, y, color='purple')
        ax.set_title("ECDF variación relativa")
        ax.set_xlabel("Variación relativa")
        ax.set_ylabel("Proporción acumulada")
        ax.grid(True)

    def plot_heatmap_tail(self):
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.tail, cmap="viridis")
        plt.title("Evolución de medias (ventana final)")
        plt.xlabel("Brazo")
        plt.ylabel("Iteración")
        plt.show()

    def plot_scatter_media_vs_variacion(self, ax):
        ax.scatter(self.mediaFinal, self.variaciones_relativas, alpha=0.6)
        ax.set_title("Media final vs variación relativa")
        ax.set_xlabel("Media final")
        ax.set_ylabel("Variación relativa")
        ax.grid(True)

    def plot_effective_metrics(self):
        mean_slate = self.resultado.trazas["mean_slate_history"]
        best_eff = self.resultado.trazas["best_arm_effective_history"]
    
        plt.figure(figsize=(12,6))
        plt.plot(mean_slate, label="Media efectiva del slate")
        plt.plot(best_eff, label="Media efectiva del mejor brazo")
        plt.title(f"Evolución de medias efectivas — {self.resultado.nombre}")
        plt.xlabel("Iteración")
        plt.ylabel("Media efectiva")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    # ============================================================
    # PANEL COMPLETO
    # ============================================================
    def render(self):
        fig = plt.figure(figsize=(22, 28))
        fig.suptitle("Dashboard del Bandit", fontsize=20)

        # BLOQUE A
        self.plot_emparejamientos_por_batch(plt.subplot(5, 3, 1))
        self.plot_emparejamientos_suavizados_por_batch(plt.subplot(5, 3, 2))
        self.plot_evidencia_acumulada(plt.subplot(5, 3, 3))
        self.plot_k_por_pelicula(plt.subplot(5, 3, 4))
        self.plot_emparejamientos_por_pelicula(plt.subplot(5, 3, 5))

        # BLOQUE B
        self.plot_reward_ruidoso(plt.subplot(5, 3, 6))
        self.plot_reward_suavizado(plt.subplot(5, 3, 7))
        self.plot_reward_acumulado(plt.subplot(5, 3, 8))
        self.plot_media_acumulada(plt.subplot(5, 3, 9))
        self.plot_gradiente(plt.subplot(5, 3, 10))

        # BLOQUE C
        self.plot_ucb_mean(plt.subplot(5, 3, 11))

        # BLOQUE D
        self.plot_variacion_relativa_hist(plt.subplot(5, 3, 12))
        self.plot_variacion_relativa_boxplot(plt.subplot(5, 3, 13))
        self.plot_variacion_relativa_ecdf(plt.subplot(5, 3, 14))
        self.plot_scatter_media_vs_variacion(plt.subplot(5, 3, 15))

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # Heatmap aparte
        self.plot_heatmap_tail()
        
        self.plot_effective_metrics()

