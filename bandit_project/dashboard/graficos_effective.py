# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 20:30:33 2026

@author: Carlos
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Media efectiva promedio del slate
# ============================================================

def plot_mean_slate_history(resultados):
    plt.figure(figsize=(12,6))

    for nombre, resultado in resultados.items():
        mean_slate = np.array(resultado.trazas["mean_slate_history"])
        plt.plot(mean_slate, label=nombre)

    plt.title("Media efectiva promedio del slate")
    plt.xlabel("Iteración")
    plt.ylabel("Media efectiva")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Media efectiva del mejor brazo estimado
# ============================================================

def plot_best_arm_effective_history(resultados):
    plt.figure(figsize=(12,6))

    for nombre, resultado in resultados.items():
        best_eff = np.array(resultado.trazas["best_arm_effective_history"])
        plt.plot(best_eff, label=nombre)

    plt.title("Media efectiva del mejor brazo estimado")
    plt.xlabel("Iteración")
    plt.ylabel("Media efectiva")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
